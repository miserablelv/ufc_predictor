import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from time import sleep
import time
import numpy as np



def get_fighter_url(name, surname):
    url = f'http://www.ufcstats.com/statistics/fighters/search?query={name}+{surname}'
    response = requests.get(url)
    while response.status_code != 200:
        time.sleep(5)
        response = requests.get(fighter_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    print(soup)
    fighters_for_name = soup.find_all('a', class_='b-link b-link_style_black')

    for fighter in fighters_for_name:
        print(fighter.text)
    
    
    urls_with_name = []
    for fighter in fighters_for_name:
        urls_with_name.append(fighter['href'])
        
    url = f'http://www.ufcstats.com/statistics/fighters/search?query={surname}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    fighters_for_surname = soup.find_all('a', class_='b-link b-link_style_black')

    print(f"Fighters for surname")
    for fighter in fighters_for_surname:
        print(fighter.text)
    
    urls_with_surname = []
    for fighter in fighters_for_surname:
        urls_with_surname.append(fighter['href'])

    fighter_url = None
    for url in urls_with_name:
        if url in urls_with_surname:
            fighter_url = url
    
    return fighter_url


def get_previous_fights(fighter_soup, cutoff_date):    
    fights = fighter_soup.find_all("tr", class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click")

    previous_urls = []
    previous_dates = []
    
    for fight in fights:
        date = datetime.strptime(fight.find_all("p", class_="b-fight-details__table-text")[12].text.strip(), '%b. %d, %Y').date()
        if date < cutoff_date:
            previous_urls.append(fight.get("data-link"))
            previous_dates.append(date)
    
    return previous_urls, previous_dates


def get_fighter_name_from_url(fighter_url):
    response = requests.get(fighter_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.find("span", class_="b-content__title-highlight").text.strip()


def get_fighter_age(fighter_soup, fight_date):
        
    birth_date = fighter_soup.body.find_all(string=re.compile(r"\b[A-Z][a-z]{2}\.? \d{1,2}, \d{4}"))[0]
    try:
        birth_datetime = datetime.strptime(birth_date.strip(), "%b %d, %Y").date()
    except ValueError:
        return 0 # AFK
        # if less_than_five_fights(fighter_soup, fight_date):
        #     return 0
        # else:
            # birth_datetime_input = input(f"Please introduce the birth date for fighter {get_fighter_name(fighter_soup)} in format like 'Jan 1, 2000'")
            # birth_datetime = datetime.strptime(birth_datetime_input.strip(), "%b %d, %Y").date()

    current_age = round((fight_date-birth_datetime).days / 365.25, 2)
    return current_age

def get_fighter_name(fighter_soup):
    name = fighter_soup.find("span", class_="b-content__title-highlight").text.strip()
    return name


def less_than_five_fights(fighter_soup, fight_date):

    previous_fights = get_all_fight_dates(fighter_soup)    
    n_previous_fights = sum(date < fight_date for date in previous_fights)
    
    return (n_previous_fights < 5)

    
def get_basic_stats(fighter_soup, fighter_url, fight_date):
    stats = {}
    
    stat_items = fighter_soup.find_all('li', class_='b-list__box-list-item')

    few_fights = less_than_five_fights(fighter_soup, fight_date)

    for item in stat_items:
        title = item.find('i', class_='b-list__box-item-title')
        if title:
            title_text = title.text.strip()
            value_text = item.text.replace(title_text, '').strip()
    
            if "Height:" in title_text:
                try:
                    stats["Height"] = int(value_text.split()[0].strip("'"))*12+int(value_text.split()[1].strip('"'))
                except ValueError:
                    stats["Height"] = 0 # AFK
                    # if few_fights:
                    #     stats["Height"] = 0 # probably unreachable
                    # else:
                    #     stats["Height"] = int(input(f"Please introduce the height in inches for {get_fighter_name(fighter_soup)}"))
                    
            elif "Weight:" in title_text:
                # stats["Weight"] = int(value_text.split()[0])
                pass
            elif "Reach:" in title_text:
                try:
                    stats["Reach"] = int(value_text.strip('"'))
                except ValueError:
                    stats["Reach"] = 0 # AFK
                    # if few_fights:
                    #     stats["Reach"] = 0 # probably unreachable
                    # else:
                    #     stats["Reach"] = int(input(f"Please introduce the reach in inches for {get_fighter_name(fighter_soup)}"))
            elif "STANCE:" in title_text:
                if value_text == "Orthodox":
                    stats["Stance_0"] = 0
                    stats["Stance_1"] = 1
                elif value_text == "Southpaw":
                    stats["Stance_0"] = 1
                    stats["Stance_1"] = 0
                elif value_text == "Switch":
                    stats["Stance_0"] = 1
                    stats["Stance_1"] = 1
                else:
                    stats["Stance_0"] = '--' # AFK
                    stats["Stance_1"] = '--'
                    # if few_fights:
                    #     stats["Stance_0"] = '--' # probably unreachable
                    #     stats["Stance_1"] = '--' # probably unreachable
                    # else:
                    #     is_orthodox = input(f"Is the fighter {get_fighter_name(fighter_soup)} orthodox? Y/N")
                    #     if is_orthodox == "Y" or is_orthodox == "y":
                    #         stats["Stance_0"] = 0
                    #         stats["Stance_1"] = 1
                    #     else:
                    #         stats["Stance_0"] = 1
                    #         stats["Stance_1"] = 0
            elif "DOB:" in title_text:
                stats["Age"] = get_fighter_age(fighter_soup, fight_date)

    return stats
    

from time import sleep


def extract_specific_strikes(fight_soup, fighter_idx):
    rows = fight_soup.find_all("tbody")[2].find_all("tr")

    cols = rows[0].find_all("td")
    
    fighters = [p.text.strip() for p in cols[0].find_all("p")]
    
    fighter_stats = {fighter: {} for fighter in fighters}
    
    categories = [
        "Sig. Strikes", "Head", "Body", "Leg",
        "Distance", "Clinch", "Ground"]
    
    specific_strikes, specific_strikes_against = {}, {}
    
    fighter_idx = 0
    
    idcs = [1, 3, 4, 5, 6, 7, 8]
    
    for i, idc in enumerate(idcs):
        values = [p.text.strip() for p in cols[idc].find_all("p")]
        specific_strikes[categories[i]+'_landed'] = int(values[fighter_idx].split(' ')[0])
        specific_strikes[categories[i]+'_attempted'] = int(values[fighter_idx].split(' ')[2])
        specific_strikes_against[categories[i]+'_landed'] = int(values[1-fighter_idx].split(' ')[0])
        specific_strikes_against[categories[i]+'_attempted'] = int(values[1-fighter_idx].split(' ')[2])
    
    return specific_strikes, specific_strikes_against



def get_score_differential(scorecards_section, won_fight):
    winner_score, loser_score = 0, 0
    for judge_score in scorecards_section:
        winner_score += int(judge_score.text.split()[-1].strip('.'))
        loser_score += int(judge_score.text.split()[-3].strip('.'))
    if won_fight:
        return winner_score - loser_score
    else:
        return loser_score - winner_score

    
def get_opponent_stats(opponent_url, fight_date):
    response = requests.get(opponent_url)
    while response.status_code != 200:
        time.sleep(5)
        response = requests.get(opponent_url)
    fighter_soup = BeautifulSoup(response.text, 'html.parser')
    
    basic_stats = get_basic_stats(fighter_soup, opponent_url, fight_date)

    previous_fights = get_all_fight_dates(fighter_soup)

    recovery_time = get_recovery_time(previous_fights, fight_date)

    n_previous_fights = sum(date < fight_date for date in previous_fights)

    if n_previous_fights > 0:
        results, wins, losses, draws_ncs = get_fighter_record(fighter_soup, n_previous_fights)
        streak = get_fighter_current_streak(results)
        title_fights, indices = get_n_title_fights(fighter_soup, n_previous_fights)
        title_wins = list(np.array(results)[indices]).count('win')
    else:
        results, wins, losses, draws_ncs = None, 0, 0, 0
        streak, title_fights, title_wins = 0, 0, 0

    opponent_stats = {"age": basic_stats['Age'], "height": basic_stats['Height'], "reach": basic_stats['Reach'], "n_fights": n_previous_fights, "n_wins": wins, "streak": streak, "n_title_fights": title_fights, 'n_title_wins': title_wins, "recovery_time": recovery_time}
    
    return opponent_stats
    

def get_fighter_fight_stats(fight_url, fighter_url, fighter_soup, fight_date):
    response = requests.get(fight_url)
    while response.status_code != 200:
        time.sleep(5)
        response = requests.get(fight_url)
    fight_soup = BeautifulSoup(response.text, 'html.parser')

    fight_details_tag = fight_soup.find("section", class_="b-fight-details__section js-fight-section").text.strip()
    if fight_details_tag != "Totals":
        return None # no details on the fight

    winning_method = fight_soup.body.find(string=re.compile(r"\s*Method:\s*")).find_next().text.split()[0]

    time_fought = fight_soup.find_all("i", class_="b-fight-details__text-item")
    final_round = int(time_fought[0].text.split()[1])
    duration_final_round = time_fought[1].text.split()[1]
    
    scorecards_section = fight_soup.body.find(string=re.compile(r"\s*Details:\s*")).find_all_next("i", class_="b-fight-details__text-item")

    fight_importance = fight_soup.find("i", class_="b-fight-details__fight-title")
    title_fight = len(fight_importance.text.split()) == 4
    
    fight_sections = fight_soup.find_all("section", class_="b-fight-details__section js-fight-section")
    
    if len(fight_sections) < 2:
        return None
    fight_table = fight_sections[1].find("table")
    
    rows = fight_table.find_all("tr")
    
    if fighter_url == fight_soup.find_all("a", class_="b-link b-fight-details__person-link")[0]['href']:
        fighter_name = fight_soup.find_all("a", class_="b-link b-fight-details__person-link")[0].text # better in the higher level function (1 vs n times)
        rival_name = fight_soup.find_all("a", class_="b-link b-fight-details__person-link")[1].text
        rival_url = fight_soup.find_all("a", class_="b-link b-fight-details__person-link")[1]['href']
        fighter_idx = 0
    elif fighter_url == fight_soup.find_all("a", class_="b-link b-fight-details__person-link")[1]['href']:
        fighter_name = fight_soup.find_all("a", class_="b-link b-fight-details__person-link")[1].text
        rival_name = fight_soup.find_all("a", class_="b-link b-fight-details__person-link")[0].text
        rival_url = fight_soup.find_all("a", class_="b-link b-fight-details__person-link")[0]['href']
        fighter_idx = 1
    else:
        raise Exception("Fighter name not adequate")

    opponent_stats = get_opponent_stats(rival_url, fight_date)
    fighter_age = get_fighter_age(fighter_soup, fight_date)
    
    previous_fights = get_all_fight_dates(fighter_soup)
    recovery_time = get_recovery_time(previous_fights, fight_date)

    fighter_0_score, fighter_1_score = 0, 0

    fighter_result = fight_soup.find("a", href=fighter_url).find_previous("i").text.strip()

    won_fight, draw_fight, loss_fight, no_contest = False, False, False, False
    if fighter_result == "W":
        won_fight = True
    elif fighter_result == "L":
        loss_fight = True
    elif fighter_result == "D":
        draw_fight = True
        
    elif fighter_result == "NC":
        no_contest = True
    
    if len(scorecards_section) != 0:
        if draw_fight is False:
            score_differential = get_score_differential(scorecards_section, won_fight)
        else:
            # score_differential = int(input(f"Please, introduce score differential for {fighter_name} against {rival_name}"))
            score_differential = 0
    else:
        score_differential = 0

    interesting_rows_idxs = [1, 2, 5, 6, 8]
    
    knockdowns = int(rows[1].find_all("td")[1].find_all("p")[fighter_idx].text.strip())
    knockdowns_against = int(rows[1].find_all("td")[1].find_all("p")[1-fighter_idx].text.strip())
    
    sig_strikes_landed = int(rows[1].find_all("td")[2].find_all("p")[fighter_idx].text.strip().split()[0])
    sig_strikes_attempted = int(rows[1].find_all("td")[2].find_all("p")[fighter_idx].text.strip().split()[2])
    sig_strikes_landed_against = int(rows[1].find_all("td")[2].find_all("p")[1-fighter_idx].text.strip().split()[0])
    sig_strikes_attempted_against = int(rows[1].find_all("td")[2].find_all("p")[1-fighter_idx].text.strip().split()[2])

    takedowns_successful = int(rows[1].find_all("td")[5].find_all("p")[fighter_idx].text.strip().split()[0])
    takedowns_attempted = int(rows[1].find_all("td")[5].find_all("p")[fighter_idx].text.strip().split()[2])
    takedowns_successful_against = int(rows[1].find_all("td")[5].find_all("p")[1-fighter_idx].text.strip().split()[0])
    takedowns_attempted_against = int(rows[1].find_all("td")[5].find_all("p")[1-fighter_idx].text.strip().split()[2])

    sub_attempts = int(rows[1].find_all("td")[7].find_all("p")[fighter_idx].text.strip())
    sub_attempts_against = int(rows[1].find_all("td")[7].find_all("p")[1-fighter_idx].text.strip())

    reversals = int(rows[1].find_all("td")[8].find_all("p")[fighter_idx].text.strip())
    reversals_against = int(rows[1].find_all("td")[8].find_all("p")[1-fighter_idx].text.strip())

    control_time = rows[1].find_all("td")[9].find_all("p")[fighter_idx].text.strip().split(' ')[0]
    controlled_time = rows[1].find_all("td")[9].find_all("p")[1-fighter_idx].text.strip().split(' ')[0]
    
    if control_time == '--':
        control_time_seconds = 0
    else:
        control_time = datetime.strptime(control_time, "%M:%S").time()
        control_time_seconds = control_time.minute * 60 + control_time.second
    if controlled_time == '--':
        controlled_time_seconds = 0
    else:
        controlled_time = datetime.strptime(controlled_time, "%M:%S").time()
        controlled_time_seconds = controlled_time.minute * 60 + controlled_time.second

    specific_strikes, specific_strikes_against = extract_specific_strikes(fight_soup, fighter_idx)
        
    fighter_stats = {
        'method': winning_method,
        'score_diff': score_differential,

        'age': fighter_age,
        'recovery_time': recovery_time,
        
        'win': won_fight,
        'loss': loss_fight,
        'draw': draw_fight,
        'no_contest': no_contest,

        'opponent_stats': opponent_stats,
        
        'title_fight': title_fight,
        'scheduled_rounds': final_round,
        'time_fought': round(((final_round - 1) * 60 * 5 + int(duration_final_round.split(':')[0])*60+int(duration_final_round.split(':')[1])) / (5 * 60), 2), # 5 min units
        
        'knockdowns': knockdowns,
        'knockdowns_against': knockdowns_against,
        
        'sig_strikes_landed': sig_strikes_landed,
        'sig_strikes_attempted': sig_strikes_attempted,
        'sig_strikes_landed_against': sig_strikes_landed_against,
        'sig_strikes_attempted_against': sig_strikes_attempted_against,

        
        'specific_strikes': specific_strikes,
        'specific_strikes_against': specific_strikes_against,
        
        'takedowns_successful': takedowns_successful,
        'takedowns_attempted': takedowns_attempted,
        'takedowns_successful_against': takedowns_successful_against,
        'takedowns_attempted_against': takedowns_attempted_against,
        
        'sub_attempts': sub_attempts,
        'sub_attempts_against': sub_attempts_against,

        'reversals': reversals,
        'reversals_against': reversals_against,
        
        'control_time': control_time_seconds,
        'controlled_time': controlled_time_seconds
    }
    
    return fighter_stats


def create_dict(): 
    total_stats_dict = {
        # context
        'age': [],
        'recovery_time': [],
        'is_undefeated': True,
        
        # record
        'score_diff': 0,
        'wins': 0,
        'losses': 0,
        'ko': 0,
        'sub': 0,
        'dec': 0,
        'kod': 0,
        'subbed': 0,
        'decd': 0,
        'title_wins': 0,
        'title_losses': 0,
        'time_fought': [],
        'dec_rounds': 0,
        'n_fights': 0,
            
        
        # how good opponents
        'opponent_stats': {
                            'n_fights': [],
                            'n_wins': [],
                            'n_title_fights': [],
                            'n_title_wins': [],
                            'streak': [],
                            'age': [],
                            'recovery_time': [],
                            # 'recent_record': [],
                            },        
        # striking
        'sig_strikes_landed': [],
        'sig_strikes_attempted': [],
        'sig_strikes_landed_against': [],
        'sig_strikes_attempted_against': [],
        
        'specific_strikes': {'Head_landed': [], 'Head_attempted': [],
                             'Body_landed': [], 'Body_attempted': [],
                             'Leg_landed': [], 'Leg_attempted': [],
                             'Distance_landed': [], 'Distance_attempted': [],
                             'Clinch_landed': [], 'Clinch_attempted': [], 
                             'Ground_landed': [], 'Ground_attempted': []},
        'specific_strikes_against': {'Head_landed': [], 'Head_attempted': [],
                             'Body_landed': [], 'Body_attempted': [],
                             'Leg_landed': [], 'Leg_attempted': [],
                             'Distance_landed': [], 'Distance_attempted': [],
                             'Clinch_landed': [], 'Clinch_attempted': [], 
                             'Ground_landed': [], 'Ground_attempted': []},
        
        'kd_landed': 0,
        'kd_against': 0,

        # wrestling
        'takedowns_successful': 0,
        'takedowns_attempted': 0,
        'takedowns_successful_against': 0,
        'takedowns_attempted_against': 0,

        # ground
        'control_time': 0,
        'controlled_time': 0,
        'sub_attempts': 0,
        'sub_attempts_against': 0,
        'reversals': 0,
        'reversals_against': 0
    }
    
    return total_stats_dict

    
def update_total_stats(total_stats, fighter_fight_stats):
    if fighter_fight_stats is None:
        return
        
    for key in fighter_fight_stats:            
        if key == "method":
            if fighter_fight_stats["win"] == True:
                if fighter_fight_stats[key] == "Decision":
                    total_stats['dec'] += 1
                elif fighter_fight_stats[key] == "KO/TKO":
                    total_stats['ko'] += 1
                elif fighter_fight_stats[key] == "Submission":
                    total_stats['sub'] += 1
            elif fighter_fight_stats["loss"] == True:
                if fighter_fight_stats[key] == "Decision":
                    total_stats['decd'] += 1
                elif fighter_fight_stats[key] == "KO/TKO":
                    total_stats['kod'] += 1
                elif fighter_fight_stats[key] == "Submission":
                    total_stats['subbed'] += 1
            else:
                pass

        elif key == "win":
            if fighter_fight_stats[key]:
                total_stats['wins'] += 1
                if fighter_fight_stats['title_fight']:
                    total_stats['title_wins'] += 1
        elif key == "loss":
            total_stats['is_undefeated'] = False
            if fighter_fight_stats[key]:
                total_stats['losses'] += 1
                if fighter_fight_stats['title_fight']:
                    total_stats['title_losses'] += 1
        elif key == "draw":
            if fighter_fight_stats[key]:
                total_stats['wins'] += 0.5
                total_stats['losses'] += 0.5
                if fighter_fight_stats['title_fight']:
                    total_stats['title_wins'] += 0.5
                    total_stats['title_losses'] += 0.5


        elif key == "no_contest":
            if fighter_fight_stats[key]:
                total_stats['wins'] += 0.5
                total_stats['losses'] += 0.5
                if fighter_fight_stats['title_fight']:
                    total_stats['title_wins'] += 0.5
                    total_stats['title_losses'] += 0.5

        elif key == "scheduled_rounds":
            if fighter_fight_stats["method"] == "Decision":
                total_stats['dec_rounds'] += fighter_fight_stats["scheduled_rounds"]
                total_stats['score_diff'] += fighter_fight_stats["score_diff"]

        elif key == "title_fight" or key == "score_diff":
            pass

        elif key == "knockdowns":
            total_stats['kd_landed'] += fighter_fight_stats[key]
            
        elif key == "knockdowns_against":
            total_stats['kd_against'] += fighter_fight_stats[key]
        
        elif key == 'specific_strikes':
            for spec_key in total_stats[key]:
                total_stats[key][spec_key].append(fighter_fight_stats[key][spec_key])

        elif key == 'specific_strikes_against':
            for spec_key in total_stats[key]:
                total_stats[key][spec_key].append(fighter_fight_stats[key][spec_key])

        elif key == 'opponent_stats':
            for spec_key in total_stats['opponent_stats']:
                total_stats['opponent_stats'][spec_key].append(fighter_fight_stats['opponent_stats'][spec_key])

        
        else:
            if type(total_stats[key]) == list:
                total_stats[key].append(fighter_fight_stats[key])
            else:
                total_stats[key] += fighter_fight_stats[key]
                
    total_stats['n_fights'] += 1


def calculate_career_stats(fighter_page_soup, fighter_url, cutoff_date):
    previous_fights_urls, previous_fights_dates = get_previous_fights(fighter_page_soup, cutoff_date)
    total_stats = create_dict()

    for fight_url, fight_date in zip(previous_fights_urls, previous_fights_dates):
        fighter_fight_stats = get_fighter_fight_stats(fight_url, fighter_url, fighter_page_soup, fight_date)
        update_total_stats(total_stats, fighter_fight_stats)

    return total_stats


def get_context_stats(fighter_soup, cutoff_date):
    previous_fights = get_all_fight_dates(fighter_soup)
    context_stats = {
        'recovery_time': get_recovery_time(previous_fights, cutoff_date),
        'unbeaten_streak': get_fighter_streak(fighter_soup, cutoff_date),
    }
    return context_stats
    

def calculate_fighter_stats(fighter_url, fighter_soup, cutoff_date):
    
    career_stats = calculate_career_stats(fighter_soup, fighter_url, cutoff_date)
    basic_stats = get_basic_stats(fighter_soup, fighter_url, cutoff_date)
    context_stats = get_context_stats(fighter_soup, cutoff_date)

    return (basic_stats, context_stats, career_stats)

    
def get_rival_url(soup):
    fighter_names = soup.find_all("a", class_="b-link b-fight-details__person-link")
    if fighter_names[0]['href'] == fighter_url:
        rival_url = fighter_names[1]['href']
    else:
        rival_url = fighter_names[0]['href']
    
    return rival_url

    
def get_n_title_fights(soup, n_fights):
    fights = soup.find_all("tr", class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click")[-n_fights:]

    idcs = []
    idx = 0
    title_fights = 0
    for fight in fights:
        for img in fight.find_all("img"):
            if 'belt' in img['src']:
                title_fights += 1
                idcs.append(idx)
        idx += 1
        
    return title_fights, idcs


def get_recovery_time(all_fights, current_date):
    if len(all_fights) == 0:
        return 200
    
    previous_fight_date = None
    
    for fight_date in all_fights:
        if fight_date < current_date:
            previous_fight_date = fight_date
            break
    
    if previous_fight_date is None:
        return 100 # default if it's his debut
    
    recovery_time = (current_date - previous_fight_date).days
    
    if recovery_time == 0:
        print(current_date, previous_fight_date)
    
    return recovery_time



def get_all_fight_dates(fighter_soup):
    raw_dates = fighter_soup.body.find_all(string=re.compile(r"\b[A-Z][a-z]{2}\.? \d{1,2}, \d{4}"))
    fight_dates = raw_dates[1:]
    cleaned_dates = [datetime.strptime(date.strip(), "%b. %d, %Y").date() for date in fight_dates]
    return cleaned_dates
    

def get_fighter_record(soup, n_fights):
    results = soup.body.find_all(string=re.compile(r"\b(win|loss|draw|NC)\b", re.IGNORECASE))[-n_fights:]
    idcs = np.where(np.array(results) == "win")
    loss_idcs = np.where(np.array(results) == "loss")
    wins = results.count('win')
    losses = results.count('loss')
    draws_ncs = results.count('draw') + results.count('nc')
    return results, wins, losses, draws_ncs

def is_fighter_undefeated(soup):
    return ('loss' not in soup.body.text)

def get_fighter_streak(fighter_soup, fight_date): ## THIS OR THE NEXT ONE?
    previous_fights = get_all_fight_dates(fighter_soup)
    n_previous_fights = sum(date < fight_date for date in previous_fights)
    results, wins, losses, draws_ncs = get_fighter_record(fighter_soup, n_previous_fights)
    fighter_streak = get_fighter_current_streak(results)
    return fighter_streak

def get_fighter_current_streak(results): ## THIS OR THE PREVIOUS ONE?
    if not results:
        return 0

    results = [r.lower() for r in results]

    i = 0
    while i < len(results) and results[i] in ["draw", "nc"]:
        i += 1

    if i == len(results):
        return 0  # only draws or NCs

    # Determine direction of the streak
    direction = results[i]
    if direction == "loss":
        sign = -1
    elif direction == "win":
        sign = 1
    else:
        sign = 0

    streak = 1
    i += 1
    while i < len(results):
        if results[i] == direction or results[i] in ["draw", "nc"]:
            if results[i] == direction:
                streak += 1
        else:
            break
        i += 1

    return sign * streak


def get_fighter_winning_methods(soup):
    methods = soup.body.find_all(string=re.compile(r"\b(ko/tko|dec|sub|cnc)\b(?!\.)", re.IGNORECASE))
    return methods
    

def clean_fighter_winning_methods(methods_raw):
    cleaned = []
    
    for m in methods_raw:
        m = m.strip().upper()
        m = re.sub(r'\s+', ' ', m)
        if 'KO/TKO' in m:
            cleaned.append('KO/TKO')
        elif 'SUB' in m:
            cleaned.append('SUB')
        elif 'DEC' in m and not re.search(r'\bDEC\.', m):
            cleaned.append('DEC')
        elif 'CNC' in m:
            cleaned.append('CNC')
            
    return cleaned