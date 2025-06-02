import numpy as np
import pandas as pd
from time import sleep

def calculate_adjusted_ko_diff(row, fighter):
    results = np.array(row[f'{fighter}_career_stats_record'])
    methods = np.array(row[f'{fighter}_career_stats_method'])
    ko_outcomes = np.where(
        (methods == 'ko') & (results == 'win'), 1,
        np.where((methods == 'ko') & (results == 'loss'), -1, 0)
    )
    return weighted_average(ko_outcomes, inverse=True)

def calculate_adjusted_sub_diff(row, fighter):
    results = np.array(row[f'{fighter}_career_stats_record'])
    methods = np.array(row[f'{fighter}_career_stats_method'])
    sub_outcomes = np.where(
        (methods == 'sub') & (results == 'win'), 1,
        np.where((methods == 'sub') & (results == 'loss'), -1, 0)
    )
    return weighted_average(sub_outcomes, inverse=True)


def calculate_sub_acc_diff(row, fighter):
    results = np.array(row[f'{fighter}_career_stats_record'])
    methods = np.array(row[f'{fighter}_career_stats_method'])
    sub_attempts = np.array(row[f'{fighter}_career_stats_sub_attempts'])
    sub_attempts_against = np.array(row[f'{fighter}_career_stats_sub_attempts_against'])
    sub_outcomes = np.where(
        (methods == 'sub') & (results == 'win'), 1, 0)
        # np.where((methods == 'sub') & (results == 'loss'), -1, 0)
    sub_acc = np.divide(
        sub_outcomes,
        sub_attempts,
        out=np.zeros_like(sub_outcomes),
        where=sub_attempts != 0,
        casting='unsafe'
    )
    sub_acc_adj = weighted_average(sub_acc, inverse=True)
    sub_outcomes_against =  np.where(
        (methods == 'sub') & (results == 'loss'), 1, 0)
    sub_acc_against = np.divide(
        sub_outcomes_against,
        sub_attempts_against,
        out=np.zeros_like(sub_outcomes_against),
        where=sub_attempts_against != 0,
        casting='unsafe'
    )
    sub_acc_against_adj = weighted_average(sub_acc_against, inverse=True)
    return sub_acc_adj - sub_acc_against_adj
    

def continuous_prime_age(age, peak_age=28.5):
    diff = abs(age - peak_age)
    return max(0, 10 - (diff**2) / 4)  # exponential decline

def recovery_score(days):
    if 180 <= days <= 340:
        return 1.0
    else:
        return np.exp(-((min(abs(days - 180), abs(days - 340))) ** 2) / (2 * (30 ** 2)))


def weighted_old_opponent_score(age_list):
    if len(age_list) == 0:
        return 0
    adjusted_age = np.array([1.5**(age - 33) if age > 33 else 0 for age in age_list])
    weights = 1.2 ** np.arange(len(age_list), 0, -1)
    try:
        return np.average(adjusted_age, weights=weights)
    except ZeroDivisionError:
        return 0


def get_fighter_mean_weight(df, fighter_id):
    a_weights = df.loc[df['Fighter_A_id'] == fighter_id, 'Fighter_A_career_stats_weight']
    b_weights = df.loc[df['Fighter_B_id'] == fighter_id, 'Fighter_B_career_stats_weight']
    all_weights = pd.concat([a_weights, b_weights])
    return weighted_average(all_weights)
    

def get_fights_weight_diff(df, full_df = None):
    if full_df is None:
        weight_mean_a = []
        weight_mean_b = []
    
        for i in range(len(df)):
            weight_mean_a.append(get_fighter_mean_weight(df, df.iloc[i]['Fighter_A_id']))
            weight_mean_b.append(get_fighter_mean_weight(df, df.iloc[i]['Fighter_B_id']))
        return np.array(weight_mean_a) - np.array(weight_mean_b)

    else:
        fighter_a_id = df['Fighter_A_id'].iloc[0]
        fighter_b_id = df['Fighter_B_id'].iloc[0]
        weight_mean_a = get_fighter_mean_weight(full_df, fighter_a_id)
        weight_mean_b = get_fighter_mean_weight(full_df, fighter_b_id)
        return weight_mean_a - weight_mean_b


def previous_weight(df, fighter_id):
    previous_div_a = df.loc[df['Fighter_A_id'] == fighter_id, 'Fight_info_division']
    previous_div_b = df.loc[df['Fighter_B_id'] == fighter_id, 'Fight_info_division']
    if previous_div_b is None:
        return previous_div_a
    elif previous_div_a is None:
        return previous_div_b
    else:
        previous_div = previous_div_a.iloc[0] if previous_div_a.index[0] < previous_div_b.index[0] else previous_div_b.iloc[0]
        return previous_div

        
def weighted_average(values, inverse=False, extra_weights=10, default=0):
    if len(values) == 0 or np.all(values == 0):
        return default
    if inverse:
        weights = 1.2**(np.arange(len(values), 0, -1))
    else:
        weights = 1.2**(np.arange(len(values)))
    try:
        return np.average(values, weights=weights*np.log(extra_weights))
    except ZeroDivisionError:
        print(f"Weights {weights}")
        print(f"Extra weights {np.log(extra_weights)}")
        return 0


def previous_weight(df, fighter_id):
    previous_div_a = df.loc[df['Fighter_A_id'] == fighter_id, ['Fight_info_division']]
    previous_div_b = df.loc[df['Fighter_B_id'] == fighter_id, ['Fight_info_division']]

    if previous_div_a.empty and previous_div_b.empty:
        return None
    elif previous_div_a.empty:
        return previous_div_b.iloc[0]['Fight_info_division']
    elif previous_div_b.empty:
        return previous_div_a.iloc[0]['Fight_info_division']
    else:
        if previous_div_a.index[0] < previous_div_b.index[0]:
            return previous_div_a.iloc[0]['Fight_info_division']
        else:
            return previous_div_b.iloc[0]['Fight_info_division']


def is_moving(row, df, fighter, direction='up'):
    try:
        current_index = df[df['Fight_id'] == row['Fight_id']].index[0]
    except:
        current_index = -1
    subset = df.iloc[current_index+1:]
    prev_weight = previous_weight(subset, row[f'{fighter}_id'])
    if direction == 'up':
        if (prev_weight is None or prev_weight >= row['Fight_info_division']):
            return 0
        else:
            return row['Fight_info_division'] - prev_weight
    elif direction == 'down':
        if (prev_weight is None or prev_weight <= row['Fight_info_division']):
            return 0
        else:
            return prev_weight - row['Fight_info_division']
    else:
        raise Exception('Unknown direction in division change')


def engineer_features(df, full_df=None):

    new_df, aux_dict = {}, {}
    
    for fighter in ["Fighter_A", "Fighter_B"]:
        
        aux_dict[f'{fighter}_kd_diff'] = df.apply(
            lambda row: weighted_average(((np.array(row[f'{fighter}_career_stats_kd_landed']) - np.array(row[f'{fighter}_career_stats_kd_against'])) / np.array(row[f'{fighter}_career_stats_time_fought'])), inverse=True),
            axis=1
        )

        aux_dict[f'{fighter}_sig_strikes_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_sig_strikes_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_sig_strikes_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_sig_strikes_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_sig_strikes_attempted'], dtype=float) != 0
                ), extra_weights=np.array(row[f'{fighter}_career_stats_sig_strikes_landed'])+2, inverse=True # +2 to avoid weights equaling 0. will end up being 0 anyway
            ),
            axis=1
        )
        
        aux_dict[f'{fighter}_sig_strikes_acc_against'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_sig_strikes_landed_against'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_sig_strikes_attempted_against'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_sig_strikes_landed_against'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_sig_strikes_attempted_against'], dtype=float) != 0
                ), extra_weights=np.array(row[f'{fighter}_career_stats_sig_strikes_landed_against'])+1, inverse=True
            ),
            axis=1
        )
        
        
        aux_dict[f'{fighter}_median_striking_diff'] = df.apply( # worse?
            lambda row: np.median((np.array(row[f'{fighter}_career_stats_sig_strikes_landed']) - np.array(row[f'{fighter}_career_stats_sig_strikes_landed_against'])) / np.array(row[f'{fighter}_career_stats_time_fought'])),
            axis=1
        )
        # print(f"Adjusted median {aux_dict[f'{fighter}_median_striking_diff'].iloc[0]}")
        
        aux_dict[f'{fighter}_mean_striking_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_sig_strikes_landed']) - np.array(row[f'{fighter}_career_stats_sig_strikes_landed_against'])) / np.sum(row[f'{fighter}_career_stats_time_fought']), inverse=True),
            axis=1
        )

        # print(f"Adjusted mean {aux_dict[f'{fighter}_mean_striking_diff'].iloc[0]}")

            
        aux_dict[f'{fighter}_striking_acc_diff'] = aux_dict[f'{fighter}_sig_strikes_acc'] - aux_dict[f'{fighter}_sig_strikes_acc_against']
    
    new_df['Fight_kd_diff'] = aux_dict['Fighter_A_kd_diff'] - aux_dict['Fighter_B_kd_diff']
    
    new_df['Fight_striking_ovr_acc_diff'] = aux_dict['Fighter_A_striking_acc_diff'] - aux_dict['Fighter_B_striking_acc_diff']
    # new_df['Fight_median_striking_acc_diff'] = df['Fighter_A_median_striking_acc'] - df['Fighter_B_median_striking_acc']
    new_df['Fight_mean_striking_diff'] = aux_dict['Fighter_A_mean_striking_diff'] - aux_dict['Fighter_B_mean_striking_diff']
    new_df['Fight_median_striking_diff'] = aux_dict['Fighter_A_median_striking_diff'] - aux_dict['Fighter_B_median_striking_diff']
    # df.drop(['Fighter_A_kd_diff', 'Fighter_B_kd_diff', 'Fighter_A_striking_acc_diff', 'Fighter_B_striking_acc_diff', 'Fighter_A_mean_striking_diff', 'Fighter_B_mean_striking_diff'], axis=1)
    
    
    for fighter in ['Fighter_A', 'Fighter_B']:
    
        aux_dict[f'{fighter}_strikes_head_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_specific_strikes_Head_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_specific_strikes_against_Head_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_specific_strikes_Head_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_specific_strikes_against_Head_attempted'], dtype=float) != 0
                ), extra_weights=np.array(row[f'{fighter}_career_stats_specific_strikes_Head_landed'])+2, inverse=True
            ),
            axis=1
        )
        
        aux_dict[f'{fighter}_strikes_head_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_specific_strikes_Head_landed']) - np.array(row[f'{fighter}_career_stats_specific_strikes_against_Head_landed'])) / np.sum(row[f'{fighter}_career_stats_time_fought']), inverse=True),
            axis=1
        )
    
        aux_dict[f'{fighter}_strikes_body_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_specific_strikes_Body_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_specific_strikes_against_Body_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_specific_strikes_Body_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_specific_strikes_against_Body_attempted'], dtype=float) != 0
                ), extra_weights=np.array(row[f'{fighter}_career_stats_specific_strikes_Body_landed'])+2, inverse=True
            ),
            axis=1
        )
        
        aux_dict[f'{fighter}_strikes_body_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_specific_strikes_Body_landed']) - np.array(row[f'{fighter}_career_stats_specific_strikes_against_Body_landed'])) / np.sum(row[f'{fighter}_career_stats_time_fought']), inverse=True),
            axis=1
        )

        aux_dict[f'{fighter}_strikes_leg_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_specific_strikes_Leg_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_specific_strikes_against_Leg_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_specific_strikes_Leg_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_specific_strikes_against_Leg_attempted'], dtype=float) != 0
                ), extra_weights=np.array(row[f'{fighter}_career_stats_specific_strikes_Leg_landed'])+2, inverse=True
            ),
            axis=1
        )
        aux_dict[f'{fighter}_strikes_leg_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_specific_strikes_Leg_landed']) - np.array(row[f'{fighter}_career_stats_specific_strikes_against_Leg_landed'])) / np.sum(row[f'{fighter}_career_stats_time_fought']), inverse=True),
            axis=1
        )

        aux_dict[f'{fighter}_strikes_distance_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_specific_strikes_Distance_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_specific_strikes_against_Distance_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_specific_strikes_Distance_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_specific_strikes_against_Distance_attempted'], dtype=float) != 0
                ), extra_weights=np.array(row[f'{fighter}_career_stats_specific_strikes_Distance_landed'])+2, inverse=True
            ),
            axis=1
        )
        aux_dict[f'{fighter}_strikes_distance_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_specific_strikes_Distance_landed']) - np.array(row[f'{fighter}_career_stats_specific_strikes_against_Distance_landed'])) / np.sum(row[f'{fighter}_career_stats_time_fought']), inverse=True),
            axis=1
        )

        aux_dict[f'{fighter}_strikes_clinch_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_specific_strikes_Clinch_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_specific_strikes_against_Clinch_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_specific_strikes_Clinch_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_specific_strikes_against_Clinch_attempted'], dtype=float) != 0
                ), extra_weights=np.array(row[f'{fighter}_career_stats_specific_strikes_Clinch_landed'])+2, inverse=True
            ),
            axis=1
        )
        aux_dict[f'{fighter}_strikes_clinch_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_specific_strikes_Clinch_landed']) - np.array(row[f'{fighter}_career_stats_specific_strikes_against_Clinch_landed'])) / np.sum(row[f'{fighter}_career_stats_time_fought']), inverse=True),
            axis=1
        )

        df[f'{fighter}_strikes_ground_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_specific_strikes_Ground_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_specific_strikes_against_Ground_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_specific_strikes_Ground_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_specific_strikes_against_Ground_attempted'], dtype=float) != 0
                ), extra_weights=np.array(row[f'{fighter}_career_stats_specific_strikes_Ground_landed'])+2, inverse=True
            ),
            axis=1
        )
        aux_dict[f'{fighter}_strikes_ground_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_specific_strikes_Ground_landed']) - np.array(row[f'{fighter}_career_stats_specific_strikes_against_Ground_landed'])) / np.sum(row[f'{fighter}_career_stats_time_fought']), inverse=True),
            axis=1
        )
        
        
    
    
    new_df['Fight_strikes_head_diff'] = aux_dict['Fighter_A_strikes_head_diff'] - aux_dict['Fighter_B_strikes_head_diff']
    new_df['Fight_strikes_body_diff'] = aux_dict['Fighter_A_strikes_body_diff'] - aux_dict['Fighter_B_strikes_body_diff']
    new_df['Fight_strikes_leg_diff'] = aux_dict['Fighter_A_strikes_leg_diff'] - aux_dict['Fighter_B_strikes_leg_diff']
    new_df['Fight_strikes_distance_diff'] = aux_dict['Fighter_A_strikes_distance_diff'] - aux_dict['Fighter_B_strikes_distance_diff']
    new_df['Fight_strikes_clinch_diff'] = aux_dict['Fighter_A_strikes_clinch_diff'] - aux_dict['Fighter_B_strikes_clinch_diff']
    new_df['Fight_strikes_ground_diff'] = aux_dict['Fighter_A_strikes_ground_diff'] - aux_dict['Fighter_B_strikes_ground_diff']

    # df.drop(['Fighter_A_strikes_head_diff', 'Fighter_B_strikes_head_diff', 'Fighter_A_strikes_body_diff', 'Fighter_B_strikes_body_diff'], axis=1)
    # df.drop(['Fighter_A_strikes_leg_diff', 'Fighter_B_strikes_leg_diff', 'Fighter_A_strikes_distance_diff', 'Fighter_B_strikes_distance_diff'], axis=1)
    # df.drop(['Fighter_A_strikes_clinch_diff', 'Fighter_B_strikes_clinch_diff', 'Fighter_A_strikes_ground_diff', 'Fighter_B_strikes_ground_diff'], axis=1)

    
    
    
    for fighter in ['Fighter_A', 'Fighter_B']:
    
        aux_dict[f'{fighter}_td_diff'] = df.apply(
            lambda row: weighted_average(((np.array(row[f'{fighter}_career_stats_takedowns_successful']) - np.array(row[f'{fighter}_career_stats_takedowns_successful_against'])) / row[f'{fighter}_career_stats_time_fought']), inverse=True),
            axis=1
        )
        
        aux_dict[f'{fighter}_control_pct'] = df.apply(
            lambda row: weighted_average(np.array(row[f'{fighter}_career_stats_control_time']) / np.array(row[f'{fighter}_career_stats_time_fought']), extra_weights=np.array(row[f'{fighter}_career_stats_control_time'])/(5*60)+2, inverse=True),
            axis=1
        )
        
        aux_dict[f'{fighter}_top_pct'] = df.apply(
            lambda row: (
                weighted_average(
                    np.divide(
                        np.array(row[f'{fighter}_career_stats_control_time'], dtype=float) / (5*60),
                        np.array(row[f'{fighter}_career_stats_control_time'], dtype=float) + np.array(row[f'{fighter}_career_stats_controlled_time']),
                        out=np.zeros_like(np.array(row[f'{fighter}_career_stats_control_time'], dtype=float)),
                        where=(np.array(row[f'{fighter}_career_stats_control_time']) + np.array(row[f'{fighter}_career_stats_control_time'])) != 0
                    ), extra_weights=np.array(row[f'{fighter}_career_stats_control_time']) / (5*60) + 1, inverse=True
                ) if np.sum(np.array(row[f'{fighter}_career_stats_control_time']) + np.array(row[f'{fighter}_career_stats_controlled_time'])) != 0 else 0.2
            ) * 100,
            axis=1
        )

        aux_dict[f'{fighter}_td_acc'] = df.apply(
            lambda row: (
                weighted_average(
                    np.divide(
                        np.array(row[f'{fighter}_career_stats_takedowns_successful'], dtype=float),
                        np.array(row[f'{fighter}_career_stats_takedowns_attempted'], dtype=float),
                        out=np.zeros_like(np.array(row[f'{fighter}_career_stats_takedowns_successful'], dtype=float)),
                        where=np.array(row[f'{fighter}_career_stats_takedowns_attempted']) != 0
                    ), extra_weights=np.array(row[f'{fighter}_career_stats_takedowns_successful'])+2, inverse=True
                ) if np.sum(row[f'{fighter}_career_stats_takedowns_attempted']) != 0 else 0.2
            ) * 100,
            axis=1
        )

        aux_dict[f'{fighter}_td_acc_against'] = df.apply(
            lambda row: (
                weighted_average(
                    np.divide(
                        np.array(row[f'{fighter}_career_stats_takedowns_successful_against'], dtype=float),
                        np.array(row[f'{fighter}_career_stats_takedowns_attempted_against'], dtype=float),
                        out=np.zeros_like(np.array(row[f'{fighter}_career_stats_takedowns_successful_against'], dtype=float)),
                        where=np.array(row[f'{fighter}_career_stats_takedowns_attempted_against']) != 0
                    ), extra_weights=np.array(row[f'{fighter}_career_stats_takedowns_successful_against'])+2, inverse=True
                ) if np.sum(row[f'{fighter}_career_stats_takedowns_attempted_against']) != 0 else 0.2
            ) * 100,
            axis=1
        )
                  
        aux_dict[f'{fighter}_td_acc_diff'] = aux_dict[f'{fighter}_td_acc'] - aux_dict[f'{fighter}_td_acc_against']
        
    
        aux_dict[f'{fighter}_reversals_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_reversals']) - np.array(row[f'{fighter}_career_stats_reversals_against'])) / np.array(row[f'{fighter}_career_stats_time_fought']), inverse=True),
            axis=1
        )
    
        # aux_dict[f'{fighter}_sub_acc'] = df.apply(
        #     lambda row: (weighted_average(np.array(row[f'{fighter}_career_stats_sub']) / np.array(row[f'{fighter}_career_stats_sub_attempts'])) if row[f'{fighter}_career_stats_sub_attempts'] != 0 else 0.2) * 100,
        #     axis=1
        # )
        
        # aux_dict[f'{fighter}_sub_acc_against'] = df.apply(
        #     lambda row: (weighted_average(np.array(row[f'{fighter}_career_stats_subbed']) / np.array(row[f'{fighter}_career_stats_sub_attempts_against'])) if row[f'{fighter}_career_stats_sub_attempts_against'] != 0 else 0.1) * 100,
        #     axis=1
        # )

        aux_dict[f'{fighter}_sub_acc_diff'] = df.apply(
            lambda row: calculate_sub_acc_diff(row, fighter),
            axis=1
        )

        # aux_dict[f'{fighter}_sub_acc_diff'] = aux_dict[f'{fighter}_sub_acc'] - aux_dict[f'{fighter}_sub_acc_against']
    
    new_df['Fight_TD_per_round_diff'] = aux_dict['Fighter_A_td_diff'] - aux_dict['Fighter_B_td_diff']


    new_df['Fight_control_pct_diff'] = aux_dict['Fighter_A_control_pct'] - aux_dict['Fighter_B_control_pct']


    # df.drop(['Fighter_A_strikes_head_diff', 'Fighter_B_strikes_head_diff', 'Fighter_A_strikes_body_diff', 'Fighter_B_strikes_body_diff'], axis=1)

    new_df['Fight_top_position_pct_diff'] = aux_dict['Fighter_A_top_pct'] - aux_dict['Fighter_B_top_pct']
    # df.drop(['Fighter_A_top_pct', 'Fighter_A_top_pct', 'Fighter_A_sub_acc', 'Fighter_A_sub_acc_against'], axis=1)
    # df.drop(['Fighter_B_sub_acc', 'Fighter_B_sub_acc_against'], axis=1)


    new_df['Fight_ovr_sub_acc_diff'] = aux_dict['Fighter_A_sub_acc_diff'] - aux_dict['Fighter_A_sub_acc_diff']
    new_df['Fight_ovr_TD_acc_diff'] = aux_dict['Fighter_A_td_acc_diff'] - aux_dict['Fighter_B_td_acc_diff']
    new_df['Fight_reversals_diff'] = aux_dict['Fighter_A_reversals_diff'] - aux_dict['Fighter_B_reversals_diff']
    # df.drop(['Fighter_A_sub_acc_diff', 'Fighter_A_sub_acc_diff', 'Fighter_A_td_acc_diff', 'Fighter_B_td_acc_diff', 'Fighter_A_reversals_diff', 'Fighter_B_reversals_diff'], axis=1)

    
    aux_dict['Fighter_A_score_diff_pct'] = df.apply(
        lambda row: (row['Fighter_A_career_stats_score_diff'] / row['Fighter_A_career_stats_dec_rounds'] if row['Fighter_A_career_stats_dec_rounds'] != 0 else 0),
        axis=1
    )
    aux_dict['Fighter_B_score_diff_pct'] = df.apply(
        lambda row: (row['Fighter_B_career_stats_score_diff'] / row['Fighter_B_career_stats_dec_rounds'] if row['Fighter_B_career_stats_dec_rounds'] != 0 else 0),
        axis=1
    )

    new_df['Fight_score_diff'] = aux_dict['Fighter_A_score_diff_pct'] - aux_dict['Fighter_B_score_diff_pct']
    # df.drop(['Fighter_A_score_diff_pct', 'Fighter_B_score_diff_pct'], axis=1)
    

    for fighter in ['Fighter_A', 'Fighter_B']:
        
    #     aux_dict[f'{fighter}_recovery_score'] = df[f'{fighter}_context_stats_recovery_time'].apply(recovery_score)
    
    
    #     # df[f'{fighter}_median_recovery_diff'] = df.apply(
    #     #     lambda row: np.median(row[f'{fighter}_career_stats_recovery_time']) - np.median(row[f'{fighter}_career_stats_opponent_stats_recovery_time']),
    #     #     axis=1
    #     # )
        aux_dict[f'{fighter}_median_recovery_score_diff'] = df.apply(
            lambda row: np.median([recovery_score(d) for d in row[f'{fighter}_career_stats_recovery_time']]) -
                        np.median([recovery_score(d) for d in row[f'{fighter}_career_stats_opponent_stats_recovery_time']]),
            axis=1
        )
    
        
    #     # df[f'{fighter}_mean_recovery_diff'] = df.apply(
    #     #     lambda row: np.mean(row[f'{fighter}_career_stats_recovery_time']) - np.mean(row[f'{fighter}_career_stats_opponent_stats_recovery_time']),
    #     #     axis=1
    #     # )
        aux_dict[f'{fighter}_mean_recovery_score_diff'] = df.apply(
            lambda row: np.mean([recovery_score(d) for d in row[f'{fighter}_career_stats_recovery_time']]) -
                        np.mean([recovery_score(d) for d in row[f'{fighter}_career_stats_opponent_stats_recovery_time']]),
            axis=1
        )
    
        
    #     # df[f'{fighter}_recovery_adv_pct'] = df.apply(
    #     #     lambda row: np.sum(np.array(row[f'{fighter}_career_stats_recovery_time']) > np.array(row[f'{fighter}_career_stats_opponent_stats_recovery_time'])) / 
    #     #                       np.array(row[f'{fighter}_career_stats_n_fights']),
    #     #     axis=1
    #     # )
        # aux_dict[f'{fighter}_recovery_score_adv_pct'] = df.apply(
        #     lambda row: np.mean([
        #         recovery_score(f_r) > recovery_score(o_r)
        #         for f_r, o_r in zip(row[f'{fighter}_career_stats_recovery_time'],
        #                             row[f'{fighter}_career_stats_opponent_stats_recovery_time'])
        #     ]),
        #     axis=1
        # )

    new_df['Fight_median_historic_recovery_diff'] = aux_dict['Fighter_B_median_recovery_score_diff'] - aux_dict['Fighter_A_median_recovery_score_diff']
    new_df['Fight_mean_historic_recovery_diff'] = aux_dict['Fighter_B_mean_recovery_score_diff'] - aux_dict['Fighter_A_mean_recovery_score_diff']
    
    
        
    for fighter in ['Fighter_A', 'Fighter_B']:
    
        aux_dict[f'{fighter}_pct_old_opponents'] = df.apply(
            lambda row: len(np.where(np.array(row[f'{fighter}_career_stats_opponent_stats_age']) > 38)[0]) / len(np.where(np.array(row[f'{fighter}_career_stats_opponent_stats_age']) > 0)[0]),
            axis=1
        )
        
        aux_dict[f'{fighter}_median_age_diff'] = df.apply(
            lambda row: (
                np.median([age for age in row[f'{fighter}_career_stats_age'] if age > 0]) -
                np.median([age for age in row[f'{fighter}_career_stats_opponent_stats_age'] if age > 0])
            ),
            axis=1
        )
    
        
        aux_dict[f'{fighter}_mean_age_diff'] = df.apply(
            lambda row: (
                np.mean([age for age in row[f'{fighter}_career_stats_age'] if age > 0]) -
                np.mean([age for age in row[f'{fighter}_career_stats_opponent_stats_age'] if age > 0])
            ),
            axis=1
        )
    
        aux_dict[f'{fighter}_age_adv_pct'] = df.apply(
            lambda row: (
                np.mean([
                    age < opponent_age
                    for age, opponent_age in zip(row[f'{fighter}_career_stats_age'], row[f'{fighter}_career_stats_opponent_stats_age'])
                    if age > 0 and opponent_age > 0
                ]) * 100
            ),
            axis=1
        )
    
        aux_dict[f'{fighter}_old_opponents_score'] = df.apply(
            lambda row: weighted_old_opponent_score(row[f'{fighter}_career_stats_opponent_stats_age']),
            axis=1
        )
        # print(f"opponents age: {df[f'{fighter}_career_stats_opponent_stats_age'].iloc[0]}")
    
        aux_dict[f'{fighter}_prime_age'] = df[f'{fighter}_basic_stats_Age'].apply(continuous_prime_age)
    
    
    new_df['Fight_median_historic_age_diff'] = aux_dict['Fighter_B_median_age_diff'] - aux_dict['Fighter_A_median_age_diff']
    new_df['Fight_mean_historic_age_diff'] = aux_dict['Fighter_B_mean_age_diff'] - aux_dict['Fighter_A_mean_age_diff']
    new_df['Fight_historic_age_adv_count_diff'] = aux_dict['Fighter_B_age_adv_pct'] - aux_dict['Fighter_A_age_adv_pct']
    new_df['Fight_historic_old_opponents_diff'] = aux_dict['Fighter_B_old_opponents_score'] - aux_dict['Fighter_A_old_opponents_score']
        
    for fighter in ['Fighter_A', 'Fighter_B']:
        
        aux_dict[f'{fighter}_win_rate'] = df.apply(
            lambda row: (len(np.where(np.array(row[f'{fighter}_career_stats_record']) == 'win')) / len(row[f'{fighter}_career_stats_record']) if len(row[f'{fighter}_career_stats_record']) != 0 else 0.5) * 100,
            axis=1
        )
        aux_dict[f'{fighter}_win_rate'] = df.apply(
            lambda row: weighted_average(np.where( np.array(row[f'{fighter}_career_stats_record']) == 'win', 1, 0), inverse=True),
            axis=1
        )
        
        
    finishing_rate, finished_rate, finishing_diff = {}, {}, {}
    
    for fighter in ["Fighter_A", "Fighter_B"]:
        
        aux_dict[f'{fighter}_ko_rate_diff'] = df.apply(
            lambda row: calculate_adjusted_ko_diff(row, fighter),
            axis=1
        )

        aux_dict[f'{fighter}_sub_rate_diff'] = df.apply(
            lambda row: calculate_adjusted_sub_diff(row, fighter),
            axis=1
        )


    new_df['Fight_finishing_ko_diff'] = aux_dict['Fighter_A_ko_rate_diff'] - aux_dict['Fighter_B_ko_rate_diff']

    new_df['Fight_finishing_sub_diff'] = aux_dict['Fighter_A_sub_rate_diff'] - aux_dict['Fighter_B_sub_rate_diff']
        

    new_df['Fight_WR_diff'] = aux_dict['Fighter_A_win_rate'] - aux_dict['Fighter_B_win_rate']

    for fighter in ["Fighter_A", "Fighter_B"]:
    
        

        aux_dict[f'{fighter}_opponents_WR'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_opponent_stats_n_wins'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_opponent_stats_n_fights'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_opponent_stats_n_wins'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_opponent_stats_n_fights'], dtype=float) != 0
                ), extra_weights=np.array(row[f'{fighter}_career_stats_opponent_stats_n_wins'])+2, inverse=True
            ),
            axis=1
        )

        aux_dict[f'{fighter}_pct_opponents_title_wins'] = df.apply(
            lambda row: weighted_average(row[f'{fighter}_career_stats_opponent_stats_n_title_wins'], inverse=True),
            axis=1
        )

        # print(f"{fighter} opponents title wins {df[f'{fighter}_career_stats_opponent_stats_n_title_wins'].iloc[0]}")
        
    new_df['Fight_opponents_WR_diff'] = aux_dict['Fighter_A_opponents_WR'] - aux_dict['Fighter_B_opponents_WR']
    new_df['Fight_opponents_title_wins_diff'] = aux_dict['Fighter_A_pct_opponents_title_wins'] - aux_dict['Fighter_B_pct_opponents_title_wins']


    for fighter in ['Fighter_A', 'Fighter_B']:
        aux_dict[f'{fighter}_young'] = df.apply(
            lambda row: 1 if float(row[f'{fighter}_basic_stats_Age']) < 25 else 0,
            axis=1
        )
        aux_dict[f'{fighter}_prime_young'] = df.apply(
            lambda row: 1 if 25 <= float(row[f'{fighter}_basic_stats_Age']) < 30 else 0,
            axis=1
        )
        aux_dict[f'{fighter}_prime'] = df.apply(
            lambda row: 1 if 30 <= float(row[f'{fighter}_basic_stats_Age']) < 33 else 0,
            axis=1
        )
        aux_dict[f'{fighter}_prime_old'] = df.apply(
            lambda row: 1 if 33 <= float(row[f'{fighter}_basic_stats_Age']) < 35 else 0,
            axis=1
        )
        aux_dict[f'{fighter}_declining'] = df.apply(
            lambda row: 1 if 35 <= float(row[f'{fighter}_basic_stats_Age']) < 37 else 0,
            axis=1
        )
        aux_dict[f'{fighter}_veteran'] = df.apply(
            lambda row: 1 if 37 <= float(row[f'{fighter}_basic_stats_Age']) <= 38.5 else 0,
            axis=1
        )
        aux_dict[f'{fighter}_ancient'] = df.apply(
            lambda row: 1 if float(row[f'{fighter}_basic_stats_Age']) > 38.5 else 0,
            axis=1
        )

    new_df['Fight_young_diff'] = aux_dict['Fighter_A_young'] - aux_dict['Fighter_B_young']
    new_df['Fight_prime_young_diff'] = aux_dict['Fighter_A_prime_young'] - aux_dict['Fighter_B_prime_young']
    new_df['Fight_prime_diff'] = aux_dict['Fighter_A_prime'] - aux_dict['Fighter_B_prime']
    new_df['Fight_prime_old_diff'] = aux_dict['Fighter_A_prime_old'] - aux_dict['Fighter_B_prime_old']
    new_df['Fight_declining_diff'] = aux_dict['Fighter_A_declining'] - aux_dict['Fighter_B_declining']
    new_df['Fight_veteran_diff'] = aux_dict['Fighter_A_veteran'] - aux_dict['Fighter_B_veteran']
    new_df['Fight_ancient_diff'] = aux_dict['Fighter_A_ancient'] - aux_dict['Fighter_B_ancient']

    # print(df['Fighter_A_basic_stats_Age'].iloc[0], df['Fighter_B_basic_stats_Age'].iloc[0])


    for fighter in ['Fighter_A', 'Fighter_B']:
        aux_dict[f'{fighter}_very_short_recovery'] = df.apply(
            lambda row: 1 if row[f'{fighter}_context_stats_recovery_time'] in (1, 30) else 0,
            axis=1
        )
        aux_dict[f'{fighter}_short_recovery'] = df.apply(
            lambda row: 1 if row[f'{fighter}_context_stats_recovery_time'] in (30, 90) else 0,
            axis=1
        )
        aux_dict[f'{fighter}_moderate_recovery'] = df.apply(
            lambda row: 1 if row[f'{fighter}_context_stats_recovery_time'] in (90, 150) else 0,
            axis=1
        )
        aux_dict[f'{fighter}_long_recovery'] = df.apply(
            lambda row: 1 if row[f'{fighter}_context_stats_recovery_time'] in (150, 250) else 0,
            axis=1
        )
        aux_dict[f'{fighter}_very_long_recovery'] = df.apply(
            lambda row: 1 if row[f'{fighter}_context_stats_recovery_time'] in (250, 375) else 0,
            axis=1
        )
        aux_dict[f'{fighter}_ring_rust'] = df.apply(
            lambda row: 1 if row[f'{fighter}_context_stats_recovery_time'] > 375 else 0,
            axis=1
        )


    new_df['Fight_very_short_recovery_diff'] = aux_dict['Fighter_A_very_short_recovery'] - aux_dict['Fighter_B_very_short_recovery']
    new_df['Fight_short_recovery_diff'] = aux_dict['Fighter_A_short_recovery'] - aux_dict['Fighter_B_short_recovery']
    new_df['Fight_short_moderate_recovery_diff'] = aux_dict['Fighter_A_moderate_recovery'] - aux_dict['Fighter_B_moderate_recovery']
    new_df['Fight_long_recovery_diff'] = aux_dict['Fighter_A_long_recovery'] - aux_dict['Fighter_B_long_recovery']
    new_df['Fight_very_long_recovery_diff'] = aux_dict['Fighter_A_very_long_recovery'] - aux_dict['Fighter_B_very_long_recovery']
    new_df['Fight_ring_rust_diff'] = aux_dict['Fighter_A_ring_rust'] - aux_dict['Fighter_B_ring_rust']
    
    
    new_df['Fight_age_diff'] = df['Fighter_B_basic_stats_Age'] - df['Fighter_A_basic_stats_Age']
    new_df['Fight_undefeated_diff'] = df['Fighter_A_career_stats_is_undefeated'].astype(int) - df['Fighter_B_career_stats_is_undefeated'].astype(int)
    # new_df['Fight_recovery_diff'] = df['Fighter_A_recovery_score'] - df['Fighter_B_recovery_score']
    # new_df['Fight_prime_age_diff'] = df['Fighter_A_prime_age'] - df['Fighter_B_prime_age']
    new_df['Fight_reach_diff'] = df['Fighter_A_basic_stats_Reach'] - df['Fighter_B_basic_stats_Reach']
    new_df['Fight_height_diff'] = df['Fighter_A_basic_stats_Height'] - df['Fighter_B_basic_stats_Height']
    new_df['Fight_stance_0_diff'] = df['Fighter_A_basic_stats_Stance_0'] - df['Fighter_B_basic_stats_Stance_0']
    new_df['Fight_stance_1_diff'] = df['Fighter_A_basic_stats_Stance_1'] - df['Fighter_B_basic_stats_Stance_1']
    new_df['Fight_streak_diff'] = df['Fighter_A_context_stats_unbeaten_streak'] - df['Fighter_B_context_stats_unbeaten_streak']

    
    for fighter in ["Fighter_A", "Fighter_B"]:
        
        if full_df is None:
            aux_dict[f'{fighter}_moving_up'] = df.apply(
                lambda row: is_moving(row, df, fighter, 'up'),
                axis=1
            )
            
            aux_dict[f'{fighter}_moving_down'] = df.apply(
                lambda row: is_moving(row, df, fighter, 'down'),
                axis=1
            )
        else:
            aux_dict[f'{fighter}_moving_up'] = df.apply(
                lambda row: is_moving(df.iloc[0], full_df, fighter, 'up'),
                axis=1
            )
            
            aux_dict[f'{fighter}_moving_down'] = df.apply(
                lambda row: is_moving(df.iloc[0], full_df, fighter, 'down'),
                axis=1
            )

    new_df['Fight_moving_up_diff'] = aux_dict['Fighter_A_moving_up'] - aux_dict['Fighter_B_moving_up'] # oversimplification? no anynmore?
    new_df['Fight_moving_down_diff'] = aux_dict['Fighter_A_moving_down'] - aux_dict['Fighter_B_moving_down']


    new_df = pd.DataFrame.from_dict(new_df)
    return new_df