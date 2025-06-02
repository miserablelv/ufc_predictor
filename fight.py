import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import time
import pandas as pd
import json
import os

from fighter import calculate_fighter_stats, get_all_fight_dates, less_than_five_fights, get_weight_class



def encode_weight_class(weight_class):
    weight_order = {
        'Flyweight': 1,
        'Bantamweight': 2,
        'Featherweight': 3,
        'Lightweight': 4,
        'Welterweight': 5,
        'Middleweight': 6,
        'Light Heavyweight': 7,
        'Heavyweight': 8,
    }
    
    return weight_order.get(weight_class.strip(), 0)


def is_women_fight(soup):
    return ("Women's" in soup.body.text)


def get_n_rounds(soup):
    n_rounds = int(soup.find_all("i", class_="b-fight-details__text-item")[2].text.split()[2])
    return n_rounds


def is_there_winner(soup):
    winner_section = soup.find("i", class_="b-fight-details__person-status b-fight-details__person-status_style_gray")
    print(winner_section.text)


def get_winner(soup):
    persons = soup.select('.b-fight-details__person')
    if len(persons) != 2:
        return None 

    fighter_a_name = persons[0].select_one('.b-fight-details__person-name').text.strip()
    fighter_b_name = persons[1].select_one('.b-fight-details__person-name').text.strip()

    fighter_a_status = persons[0].select_one('i').text.strip()
    fighter_b_status = persons[1].select_one('i').text.strip()

    if fighter_a_status == 'W':
        return 'A', fighter_a_name
    elif fighter_b_status == 'W':
        return 'B', fighter_b_name
    else:
        return 'Draw/NC', None


def get_fight_details(soup):
    weight_class = encode_weight_class(get_weight_class(soup))
    n_rounds = get_n_rounds(soup)
    winner = get_winner(soup)
    fight_details = {'division': weight_class, 'n_rounds': n_rounds}
    return fight_details    


def new_fight_dict(): # necessary?
    fight_dict = {'Fight_info': None,
                  'Fighter_A_basic_stats': None,
                  'Fighter_B_basic_stats': None,
                  'Fighter_A_context_stats': None,
                  'Fighter_B_context_stats': None,
                  'Fighter_A_career_stats': None,
                  'Fighter_B_career_stats': None}
    return fight_dict


def get_fighters(soup):
    fighters = [soup.find_all("a", class_="b-link b-fight-details__person-link")[i]['href'] for i in range(2)]    
    return fighters


def fight_already_fetched(fight_link, df): # should also check if every feature is present
    # if df is None:
        # return False
    try:
        fight_id = fight_link.split('/')[4]
        return (df["Fight_id"].str.contains(fight_id).any())
    except KeyError:
        return False

def get_all_events_urls():
    url = 'http://www.ufcstats.com/statistics/events/completed?page=all'

    response = requests.get(url)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    event_links = soup.find_all("a", class_="b-link b-link_style_black")
    
    event_urls = [event_link['href'] for event_link in event_links]

    return event_urls



def get_hypothetical_fight(fighter_0, fighter_1, fight_date):                
    fighters = [fighter_0, fighter_1] 

    fighters_soup = []

    for fighter in fighters:
        response = requests.get(fighter)
        while response.status_code != 200:
            time.sleep(5)
            response = requests.get(fighter)
        fighter_soup = BeautifulSoup(response.text, 'html.parser')
        fighters_soup.append(fighter_soup)
            
    fight_dict = new_fight_dict()
    fight_dict['Fighter_A_id'] = fighters[0].split('/')[4]
    fight_dict['Fighter_B_id'] = fighters[1].split('/')[4]
    
    fight_dict['Fighter_A_basic_stats'],  fight_dict['Fighter_A_context_stats'],  fight_dict['Fighter_A_career_stats'] = calculate_fighter_stats(fighters[0], fighters_soup[0], fight_date)
    fight_dict['Fighter_B_basic_stats'], fight_dict['Fighter_B_context_stats'],  fight_dict['Fighter_B_career_stats'] = calculate_fighter_stats(fighters[1], fighters_soup[1], fight_date)

    
    flat_fight = flatten_fight_data(fight_dict)

    fight_df = pd.DataFrame([flat_fight])

    ## save hypothetical fights in a separate df
    # if not os.path.exists("hypothetical_fights.json"):
    #     with open("hypothetical_fights.json", 'w') as f:
    #         json.dump([flat_fight], f, indent=4)
    # else:
    #     with open("hypothetical_fights.json", 'r+') as f:
    #         # print("Saving fight data...")
    #         data = json.load(f)
    #         data.append(flat_fight)
    #         f.seek(0)
    #         json.dump(data, f, indent=4)

    return fight_df

    

def extract_fights_in_events(events_urls): # it would be better to return 1 by 1
    fights_X, fights_y = [], []
    try:
        with open("fights_df.json", 'r+') as f:
            data = json.load(f)
            current_df = pd.DataFrame(data)
    except: 
        current_df = None
    
    for event in events_urls:
        start = time.time()
        
        response = requests.get(event)
        soup =  BeautifulSoup(response.text, 'html.parser')
     
        event_title = soup.find_all("span", class_="b-content__title-highlight")[0].text.strip()
        event_date_tag = soup.find("i", string=re.compile(r"\sDate:\s*", re.IGNORECASE))
        event_date = event_date_tag.find_parent().text.replace("Date:", "").strip()
        event_datetime = datetime.strptime(event_date, '%B %d, %Y').date()
        
        print(f"Extracting data from {event_title} on {event_date}")
    
        fight_rows = soup.find_all("tr", class_="b-fight-details__table-row__hover")
        
        fight_links = [row.get("data-link") for row in fight_rows if row.get("data-link")]
        
        for fight_link in fight_links:
            fight_dict = new_fight_dict()
            response = requests.get(fight_link)
            soup = BeautifulSoup(response.text, 'html.parser')

            if current_df is not None and fight_already_fetched(fight_link, current_df):
                continue
            
            if is_women_fight(soup):
                continue
                
            fighters = get_fighters(soup) # get the urls

            fighters_soup = []

            for fighter in fighters:
                response = requests.get(fighter)
                while response.status_code != 200:
                    time.sleep(5)
                    response = requests.get(fighter)
                fighter_soup = BeautifulSoup(response.text, 'html.parser')
                fighters_soup.append(fighter_soup)
                    
            if less_than_five_fights(fighters_soup[0], event_datetime) or less_than_five_fights(fighters_soup[1], event_datetime):
                continue

            fight_dict['Fight_info'] = get_fight_details(soup)
            
            fight_dict['Fight_id'] = fight_link.split('/')[4]
            fight_dict['Fighter_A_id'] = fighters[0].split('/')[4]
            fight_dict['Fighter_B_id'] = fighters[1].split('/')[4]
            
            fight_dict['Fighter_A_basic_stats'],  fight_dict['Fighter_A_context_stats'],  fight_dict['Fighter_A_career_stats'] = calculate_fighter_stats(fighters[0], fighters_soup[0], event_datetime)
            fight_dict['Fighter_B_basic_stats'], fight_dict['Fighter_B_context_stats'],  fight_dict['Fighter_B_career_stats'] = calculate_fighter_stats(fighters[1], fighters_soup[1], event_datetime)

            
            flat_fight = flatten_fight_data(fight_dict)
    
            if not os.path.exists("fights_df.json"):
                with open("fights_df.json", 'w') as f:
                    json.dump([flat_fight], f, indent=4)
            else:
                with open("fights_df.json", 'r+') as f:
                    data = json.load(f)
                    data.append(flat_fight)
                    f.seek(0)
                    json.dump(data, f, indent=4)
        
            result = get_winner(soup)[0]
            fight_result = {"id": fight_dict["Fight_id"], "winner": result}
        
            if not os.path.exists("fight_results.json"):
                with open("fight_results.json", 'w') as f:
                    json.dump([fight_result], f, indent=4)
            else:
                with open("fight_results.json", 'r+') as f:
                    data = json.load(f)
                    data.append(fight_result)
                    f.seek(0)
                    json.dump(data, f, indent=4)

        end = time.time()
        print(f"Time taken to extract event is {end - start}")
    
    return fights_X, fights_y


def flatten_fight_data(fight_data):
    flat_data = {}

    for section_key, section_val in fight_data.items():
        if isinstance(section_val, dict):
            for key, value in section_val.items():
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        flat_data[f"{section_key}_{key}_{subkey}"] = subval
                else:
                    flat_data[f"{section_key}_{key}"] = value
        else:
            flat_data[section_key] = section_val

    return flat_data