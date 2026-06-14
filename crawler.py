import re
from bs4 import BeautifulSoup
import wikipedia
from urllib.request import Request, urlopen


def get_movie_titles(wiki_url) -> set :

    wikipedia.set_lang("fr")
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = Request(wiki_url, headers = hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, 'html.parser')
    
    tables = soup.find_all('table', {'class': 'wikitable'})
    
    movie_titles = set() 
    
    for table in tables[:2]: # la page contient 2 tables
        rows = table.find_all('tr')
        for row in rows[1:]: # Ignorer les en-têtes
            cols = row.find_all(['td', 'th'])
            if len(cols) > 1:
                title_cell = cols[1]
                title = title_cell.get_text(strip=True)

                year_cell = cols[3]
                year = re.sub(r'\[.*?\]', '', year_cell.get_text(strip=True)).strip()
                
                clean_title = re.sub(r'\[.*?\]', '', title).strip() 
                if clean_title :
                    movie_titles.add((clean_title, year))
    seen = {}
    for title, year in movie_titles:
        if title not in seen:
            seen[title] = year
    return list(seen.items())
    return [(t, y) for t, y in list(seen.items()) if t in ["Jeanne d'Arc", "Lucy", "La Sorcière", "L'Animal", "Fantomas se déchaîne"]]

def load_movies_urls(wiki_url: str, agent: str) -> tuple[list[str], list[str]]:
    import time
    
    movie_titles = get_movie_titles(wiki_url)
                    
    print(f"{len(movie_titles)} films uniques trouvés. Interrogation de l'API Wikipedia...")

    wikipedia.set_user_agent(agent)
    
    movie_urls = []
    available_titles = []
    i = 0
    for title, year in movie_titles:
        i+=1
        try:
            #On teste d'abord l'url avec "*titre* (film, *année*)"" pour éviter les cas d'homonymie (ex: Jeanne d'Arc, pour éviter de tomber sur la page du personnage historique ou d'une autre oeuvre)
            # auto_suggest=False évite de se retrouver sur une page inattendue si le titre est très court
            page = wikipedia.page(f"{title} (film, {year})", auto_suggest=False)
            movie_urls.append(page.url)
            available_titles.append(title)
        except wikipedia.exceptions.PageError:
            # Deuxième cas d'homonymie : on ajoute simplement " (film)" au titre (ex: "L'Animal")
            try:
                page = wikipedia.page(f"{title} (film)", auto_suggest=False)
                movie_urls.append(page.url)
                available_titles.append(title)
            except wikipedia.exceptions.PageError:
                #Pour finir si le titre du film est une page unique, on essaie simplement avec celui-ci (ex: "Fantomas se déchaîne")
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    movie_urls.append(page.url)
                    available_titles.append(title)
                except wikipedia.exceptions.PageError:
                    print(f"[SKIP] film introuvable : {title} ({year})")
                
        except wikipedia.exceptions.PageError:
            # La page n'existe pas avec ce titre exact
            print(f"[SKIP] film introuvable : {title} ({year})")
        if (i%10 ==0):
            time.sleep(1)
        #if i >9:
            #return movie_urls, available_titles
            
    #print(len(movie_urls))
    return movie_urls, available_titles