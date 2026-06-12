import requests
import re
from urllib.parse import unquote
from bs4 import BeautifulSoup, Tag

SECTIONS_TEXT_TO_KEEP = {'synopsis', 'accueil', 'analyse', 'réception', 'reception'}
SECTIONS_SQL_TO_KEEP  = {'fiche technique', 'distribution', 'distinctions'}

def clean_wiki_refs(text: str) -> str | None :
    """Supprime les références Wikipédia du type [1], [2], etc."""
    if not text:
        return 
    # Enlève les références [...]
    cleaned = re.sub(r'\[.*?\]', '', text)
    # Supprime les espaces multiples créés par la suppression
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def is_h2(elem: Tag) -> bool:
    """ Vérifie qu'il s'agit d'une balise h2 selon les différent formats :
        * < h2 >
        * < div class="mw-heading mw-heading2" > (Wikipedia 2023+)
    """
    if elem.name == 'h2':
        return True
    classes = elem.get('class')
    return elem.name == 'div' and isinstance(classes, list) and 'mw-heading2' in classes


def is_subheading(elem: Tag) -> bool:
    """Vérifie qu'il s'agit d'une balise h3/h4 selon les différent formats :
        * < h3 >
        * < div class="mw-heading mw-heading3" > (Wikipedia 2023+)
    Utilisé pour parser les sous-catégories des sections (ex. [César], [BAFTA], [Lumières]…).
    """
    if elem.name in ('h3', 'h4'):
        return True
    classes = elem.get('class') # Nouveau format Wikipedia 2023+ : <div class="mw-heading mw-heading3">
    return elem.name == 'div' and isinstance(classes, list) and any(c in ('mw-heading3', 'mw-heading4') for c in classes)


def heading_text(elem: Tag) -> str:
    """
    Renvoie le texte de la balise <h2> associé à elem.
    Si elem ne contient pas de balise h2, renvoie le texte de elem directement.
    Gère les deux formats Wikipedia :
    - Ancien : <h2><span class="mw-headline">Titre</span><span class="mw-editsection">...</span></h2>
    - Nouveau : <div class="mw-heading2"><h2>Titre</h2><span class="mw-editsection">...</span></div>
    """
    h2 = elem if elem.name == 'h2' else elem.find('h2')
    if h2 is not None:
        headline = h2.find('span', class_='mw-headline')
        if headline:
            return headline.get_text(strip=True)
        return h2.get_text(strip=True)
    return elem.get_text(strip=True)


def push_section(result: list[str], title: str | None, paras: list[str]) -> None:
    """Ajoute une section formatée à result si elle doit être conservée."""
    if not paras:
        return
    if title is None:
        # On nomme la 1ère partie de la page "Introduction"
        result.append("Introduction\n" + "\n".join(paras))
    elif title.lower() in SECTIONS_TEXT_TO_KEEP:
        result.append(f"{title}\n" + "\n".join(paras))

def flat_elems(parent: Tag):
    """Itère les enfants en dépliant les wrappers <section> de Wikipedia 2024+."""
    for child in parent.children:
        if not isinstance(child, Tag):
            continue
        if child.name in ('section', 'meta'):
            yield from flat_elems(child)
        else:
            yield child

def extract_movie_sections(url: str) -> tuple[str, dict]:
    """Charge la page Wikipedia une seule fois et retourne un tuple :
    - [0] str  : intro + sections textuelles (pour la base vectorielle)
    - [1] dict : titre, fiche technique, distribution, distinctions
    """
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')

    # Titre de la page, utilisé comme contexte pour chaque chunk indexé (cf. retriever.py)
    h1 = soup.find('h1', {'id': 'firstHeading'})
    titre = h1.get_text(strip=True) if h1 else unquote(url.rsplit('/', 1)[-1]).replace('_', ' ')

    parser_output = soup.find('div', {'class': 'mw-parser-output'})
    if not parser_output:
        return "", {'url': url, 'titre': titre, 'fiche_technique': {}, 'distribution': [], 'distinctions': []}

    current_title: str | None = None
    text_result: list[str] = []
    text_paras: list[str] = []

    # On stocke dans un 1er temps les balises Tag brutes de chaque section SQL (pour gérer les différentes structures HTML séparément).
    sql_elems: dict[str, list[Tag]] = {}

    for elem in flat_elems(parser_output):
        if not isinstance(elem, Tag):
            continue
        
        # Nouvelle section (on push l'ancienne et on met à jour le titre)
        if is_h2(elem):
            push_section(text_result, current_title, text_paras)
            current_title = heading_text(elem)
            text_paras = []
            
        else:
            # Extraction textuelle (intro + SECTIONS_TEXT_TO_KEEP)
            if elem.name == 'p':
                text = clean_wiki_refs(elem.get_text(strip=True))
                if text:
                    text_paras.append(text)
            elif elem.name in ('ul', 'ol') and current_title is not None:
                text = clean_wiki_refs(elem.get_text(separator=' ', strip=True))
                if text:
                    text_paras.append(text)

            # Si on est dans une section SQL, on mémorise le Tag pour le traiter après.
            if current_title is not None and current_title.lower() in SECTIONS_SQL_TO_KEEP:
                sql_elems.setdefault(current_title.lower(), []).append(elem)

    push_section(text_result, current_title, text_paras)

    # On convertit ensuite les Tags bruts en structures exploitables (selon si c'est une liste ou une table)
    # - Liste <ul>/<li> avec items "Clé : Valeur" (format le plus courant)
    # - Table <wikitable> avec <th> (clé) et <td> (valeur)
    
    fiche: dict[str, str] = {}
    for elem in sql_elems.get('fiche technique', []):
        if elem.name in ('ul', 'ol'):
            for li in elem.find_all('li', recursive=False):
                text = clean_wiki_refs(li.get_text(separator=' ', strip=True))
                if text and ' : ' in text:
                    key, _, value = text.partition(' : ')
                    if key.strip() and value.strip():
                        fiche[key.strip()] = value.strip()
        else:
            table = elem if elem.name == 'table' else elem.find('table')
            if not isinstance(table, Tag):
                continue
            for row in table.find_all('tr'):
                th = row.find('th')
                td = row.find('td')
                if th and td:
                    key = clean_wiki_refs(th.get_text(strip=True))
                    value = clean_wiki_refs(td.get_text(separator=', ', strip=True))
                    if key and value:
                        fiche[key] = value

    # Distribution : liste <ul>/<ol> d'items du type "Acteur : Rôle".
    # On produit une liste de chaînes, une par rôle.
    distribution: list[str] = []
    for elem in sql_elems.get('distribution', []):
        for li in elem.find_all('li'):
            text = clean_wiki_refs(li.get_text(strip=True))
            if text:
                distribution.append(text)

    # Distinctions : section organisée en sous-catégories (César, BAFTA, Lumières…)
    # signalées par des h3/h4, suivies de listes d'items (nominations et prix remportés).
    # On préfixe chaque sous-titre entre crochets pour garder le contexte dans la liste plate.
    # Exemple : ["[César]", "Meilleur film", "Meilleur réalisateur", "[BAFTA]", …]
    distinctions: list[str] = []
    for elem in sql_elems.get('distinctions', []):
        if is_subheading(elem):
            label = clean_wiki_refs(elem.get_text(strip=True))
            if label:
                distinctions.append(f"[{label}]")
        elif elem.name in ('ul', 'ol'):
            for li in elem.find_all('li'):
                text = clean_wiki_refs(li.get_text(strip=True))
                if text:
                    distinctions.append(text)
        elif elem.name == 'p':
            text = clean_wiki_refs(elem.get_text(strip=True))
            if text:
                distinctions.append(text)

    return (
        "\n\n".join(text_result),
        {'url': url, 'titre': titre, 'fiche_technique': fiche, 'distribution': distribution, 'distinctions': distinctions},
    )