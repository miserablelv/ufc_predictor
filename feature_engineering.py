import numpy as np
import pandas as pd




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
    weights = [np.exp((age - 33) / 2) if age > 33 else 0 for age in age_list]
    return np.sum(weights) / len(age_list)


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
            fighter_id = df.iloc[i]['Fighter_A_id']
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
    print(previous_div_a, previous_div_b)
    if previous_div_b is None:
        return previous_div_a
    elif previous_div_a is None:
        return previous_div_b
    else:
        previous_div = previous_div_a.iloc[0] if previous_div_a.index[0] < previous_div_b.index[0] else previous_div_b.iloc[0]
        print(previous_div)
        return previous_div

        
def weighted_average(values):
    weights = np.arange(len(values), 0, -1)
    return np.average(values, weights=weights)


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
        # Compare their original indices to get the earlier one
        if previous_div_a.index[0] < previous_div_b.index[0]:
            return previous_div_a.iloc[0]['Fight_info_division']
        else:
            return previous_div_b.iloc[0]['Fight_info_division']


def is_moving_up(row, df, fighter):
    try:
        current_index = df[df['Fight_id'] == row['Fight_id']].index[0]
    except:
        current_index = -1
    subset = df.iloc[current_index+1:]
    prev_weight = previous_weight(subset, row[f'{fighter}_id'])
    if prev_weight is None or prev_weight >= row['Fight_info_division']:
        return 0
    # print(f"Prev weight for {fighter}: {prev_weight}, new weight {row['Fight_info_division']}")
    return row['Fight_info_division'] - prev_weight

def is_moving_down(row, df, fighter): # redundant?
    try:
        current_index = df[df['Fight_id'] == row['Fight_id']].index[0]
    except:
        current_index = -1
    subset = df.iloc[current_index+1:]
    prev_weight = previous_weight(subset, row[f'{fighter}_id'])
    if prev_weight is None or prev_weight <= row['Fight_info_division']:
        return 0
    # print(f"Prev weight for {fighter}: {prev_weight}, new weight {row['Fight_info_division']}")
    return prev_weight - row['Fight_info_division']


def engineer_features(df, full_df=None):
# for df in [real_df, hypothetical_df]:

    new_df, aux_dict = {}, {}
    
    
    for fighter in ["Fighter_A", "Fighter_B"]:
    
        aux_dict[f'{fighter}_kd_diff'] = df.apply(
            lambda row: (np.sum(row[f'{fighter}_career_stats_kd_landed']) - np.sum(row[f'{fighter}_career_stats_kd_against'])) / np.sum(row[f'{fighter}_career_stats_time_fought']),
            axis=1
        )

        aux_dict[f'{fighter}_sig_strikes_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_sig_strikes_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_sig_strikes_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_sig_strikes_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_sig_strikes_attempted'], dtype=float) != 0
                )
            ),
            axis=1
        )
        # df[f'{fighter}_sig_strikes_acc'] = df.apply(
        #     lambda row: weighted_average(np.array(row[f'{fighter}_career_stats_sig_strikes_landed']) / np.array(row[f'{fighter}_career_stats_sig_strikes_attempted'])), #if np.sum(row[f'{fighter}_career_stats_sig_strikes_attempted']) > 0 else 0.3,
        #     axis=1
        # )
        
        aux_dict[f'{fighter}_sig_strikes_acc_against'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_sig_strikes_landed_against'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_sig_strikes_attempted_against'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_sig_strikes_landed_against'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_sig_strikes_attempted_against'], dtype=float) != 0
                )
            ),
            axis=1
        )
        # df[f'{fighter}_sig_strikes_acc_against'] = df.apply(
        #     lambda row: weighted_average(np.array(row[f'{fighter}_career_stats_sig_strikes_landed_against']) / np.array(row[f'{fighter}_career_stats_sig_strikes_attempted_against'])),
        #     axis=1
        # )
    
        # df[f'{fighter}_mean_striking_acc'] = df.apply(
        #     lambda row: np.mean(row[f'{fighter}_career_stats_sig_strikes_landed']) / np.mean(row[f'{fighter}_career_stats_sig_strikes_attempted']),
        #     axis=1
        # )
        # df[f'{fighter}_mean_striking_acc'] = df.apply(
        #     lambda row: weighted_average(row[f'{fighter}_career_stats_sig_strikes_landed']) / weighted_average(row[f'{fighter}_career_stats_sig_strikes_attempted']),
        #     axis=1
        # )
        
        # df[f'{fighter}_median_striking_acc'] = df.apply(
        #     lambda row: np.median(row[f'{fighter}_career_stats_sig_strikes_landed']) / np.median(row[f'{fighter}_career_stats_sig_strikes_attempted']),#) if np.sum(row[f'{fighter}_career_stats_sig_strikes_attempted']) > 0 else 0.3) * 100,
        #     axis=1
        # )
        
        aux_dict[f'{fighter}_median_striking_diff'] = df.apply( # not accounting trend
            lambda row: np.median(np.array(row[f'{fighter}_career_stats_sig_strikes_landed']) - np.array(row[f'{fighter}_career_stats_sig_strikes_landed_against'])),
            axis=1
        )
        
        aux_dict[f'{fighter}_mean_striking_diff'] = df.apply( # accounting trend
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_sig_strikes_landed']) - np.array(row[f'{fighter}_career_stats_sig_strikes_landed_against'])) / np.sum(row[f'{fighter}_career_stats_time_fought'])),
            axis=1
        )
            
        # df[f'{fighter}_striking_acc_diff'] = df.apply(
        #     lambda row: row[f'{fighter}_sig_strikes_acc'] - row[f'{fighter}_sig_strikes_acc_against'],
        #     axis=1
        # )
        aux_dict[f'{fighter}_striking_acc_diff'] = aux_dict[f'{fighter}_sig_strikes_acc'] - aux_dict[f'{fighter}_sig_strikes_acc_against']
    
    new_df['Fight_kd_diff'] = aux_dict['Fighter_A_kd_diff'] - aux_dict['Fighter_B_kd_diff']
    
    new_df['Fight_striking_ovr_acc_diff'] = aux_dict['Fighter_A_striking_acc_diff'] - aux_dict['Fighter_B_striking_acc_diff'] # ofensive and defensive
    # new_df['Fight_mean_striking_acc_diff'] = df['Fighter_A_mean_striking_acc'] - df['Fighter_B_mean_striking_acc'] # only ofensive
    # new_df['Fight_median_striking_acc_diff'] = df['Fighter_A_median_striking_acc'] - df['Fighter_B_median_striking_acc'] # only ofensive
    new_df['Fight_mean_striking_diff'] = aux_dict['Fighter_A_mean_striking_diff'] - aux_dict['Fighter_B_mean_striking_diff']
    # new_df['Fight_median_striking_diff'] = df['Fighter_A_median_striking_diff'] - df['Fighter_B_median_striking_diff'] # maybe take out?
    # df.drop(['Fighter_A_kd_diff', 'Fighter_B_kd_diff', 'Fighter_A_striking_acc_diff', 'Fighter_B_striking_acc_diff', 'Fighter_A_mean_striking_diff', 'Fighter_B_mean_striking_diff'], axis=1)
    
    
    for fighter in ['Fighter_A', 'Fighter_B']:
    
        aux_dict[f'{fighter}_strikes_head_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_specific_strikes_Head_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_specific_strikes_against_Head_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_specific_strikes_Head_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_specific_strikes_against_Head_attempted'], dtype=float) != 0
                )
            ),
            axis=1
        )
        aux_dict[f'{fighter}_strikes_head_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_specific_strikes_Head_landed']) - np.array(row[f'{fighter}_career_stats_specific_strikes_against_Head_landed'])) / np.sum(row[f'{fighter}_career_stats_time_fought'])),
            axis=1
        )
    
    
        aux_dict[f'{fighter}_strikes_body_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_specific_strikes_Body_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_specific_strikes_against_Body_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_specific_strikes_Body_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_specific_strikes_against_Body_attempted'], dtype=float) != 0
                )
            ),
            axis=1
        )
        aux_dict[f'{fighter}_strikes_body_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_specific_strikes_Body_landed']) - np.array(row[f'{fighter}_career_stats_specific_strikes_against_Body_landed'])) / np.sum(row[f'{fighter}_career_stats_time_fought'])),
            axis=1
        )

        aux_dict[f'{fighter}_strikes_leg_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_specific_strikes_Leg_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_specific_strikes_against_Leg_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_specific_strikes_Leg_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_specific_strikes_against_Leg_attempted'], dtype=float) != 0
                )
            ),
            axis=1
        )
        aux_dict[f'{fighter}_strikes_leg_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_specific_strikes_Leg_landed']) - np.array(row[f'{fighter}_career_stats_specific_strikes_against_Leg_landed'])) / np.sum(row[f'{fighter}_career_stats_time_fought'])),
            axis=1
        )

        aux_dict[f'{fighter}_strikes_distance_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_specific_strikes_Distance_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_specific_strikes_against_Distance_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_specific_strikes_Distance_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_specific_strikes_against_Distance_attempted'], dtype=float) != 0
                )
            ),
            axis=1
        )
        aux_dict[f'{fighter}_strikes_distance_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_specific_strikes_Distance_landed']) - np.array(row[f'{fighter}_career_stats_specific_strikes_against_Distance_landed'])) / np.sum(row[f'{fighter}_career_stats_time_fought'])),
            axis=1
        )

        aux_dict[f'{fighter}_strikes_clinch_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_specific_strikes_Clinch_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_specific_strikes_against_Clinch_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_specific_strikes_Clinch_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_specific_strikes_against_Clinch_attempted'], dtype=float) != 0
                )
            ),
            axis=1
        )
        aux_dict[f'{fighter}_strikes_clinch_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_specific_strikes_Clinch_landed']) - np.array(row[f'{fighter}_career_stats_specific_strikes_against_Clinch_landed'])) / np.sum(row[f'{fighter}_career_stats_time_fought'])),
            axis=1
        )

        df[f'{fighter}_strikes_ground_acc'] = df.apply(
            lambda row: weighted_average(
                np.divide(
                    np.array(row[f'{fighter}_career_stats_specific_strikes_Ground_landed'], dtype=float),
                    np.array(row[f'{fighter}_career_stats_specific_strikes_against_Ground_attempted'], dtype=float),
                    out=np.zeros_like(np.array(row[f'{fighter}_career_stats_specific_strikes_Ground_landed'], dtype=float)),
                    where=np.array(row[f'{fighter}_career_stats_specific_strikes_against_Ground_attempted'], dtype=float) != 0
                )
            ),
            axis=1
        )
        aux_dict[f'{fighter}_strikes_ground_diff'] = df.apply(
            lambda row: weighted_average((np.array(row[f'{fighter}_career_stats_specific_strikes_Ground_landed']) - np.array(row[f'{fighter}_career_stats_specific_strikes_against_Ground_landed'])) / np.sum(row[f'{fighter}_career_stats_time_fought'])),
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
    
        # takedowns per 15 min (not using it right now because the next one is more complete)
        aux_dict[f'{fighter}_td_per_round'] = df.apply(
            lambda row: weighted_average(np.array(row[f'{fighter}_career_stats_takedowns_successful']) / np.array(row[f'{fighter}_career_stats_time_fought'])),
            axis=1
        )
        
        aux_dict[f'{fighter}_td_diff'] = df.apply(
            lambda row: (np.sum(row[f'{fighter}_career_stats_takedowns_successful']) - np.array(row[f'{fighter}_career_stats_takedowns_successful_against'])) / np.sum(row[f'{fighter}_career_stats_time_fought']),
            axis=1
        )
        
        # fighter = "Fighter_A"
        
        # control (could also do it per round instead of pct) We cannot do diff because we are not accessing the opponent's historic control time i think
        aux_dict[f'{fighter}_control_pct'] = df.apply(
            lambda row: (np.sum(row[f'{fighter}_career_stats_control_time']) / (np.sum(row[f'{fighter}_career_stats_time_fought']) * 5 * 60)) * 100,
            axis=1
        )
        
        # control
        aux_dict[f'{fighter}_top_pct'] = df.apply(
            lambda row: (np.sum(row[f'{fighter}_career_stats_control_time']) / (np.sum(row[f'{fighter}_career_stats_control_time']) + np.sum(row[f'{fighter}_career_stats_controlled_time'])) * 100),
            axis=1
        )
        
        aux_dict[f'{fighter}_td_acc'] = df.apply(
            lambda row: (row[f'{fighter}_career_stats_takedowns_successful'] / row[f'{fighter}_career_stats_takedowns_attempted'] if np.sum(row[f'{fighter}_career_stats_takedowns_attempted']) != 0 else 0.2) * 100,
            axis=1
        )
        
        aux_dict[f'{fighter}_td_acc_against'] = df.apply(
            lambda row: (row[f'{fighter}_career_stats_takedowns_successful_against'] / row[f'{fighter}_career_stats_takedowns_attempted_against'] if np.sum(row[f'{fighter}_career_stats_takedowns_attempted_against']) != 0 else 0.2) * 100,
            axis=1
        )
    
        # df[f'{fighter}_td_acc_diff'] = df.apply(
        #     lambda row: row[f'{fighter}_td_acc'] - row[f'{fighter}_td_acc_against'],
        #     axis=1
        # )
        aux_dict[f'{fighter}_td_acc_diff'] = aux_dict[f'{fighter}_td_acc'] - aux_dict[f'{fighter}_td_acc_against']
        
    
        aux_dict[f'{fighter}_reversals_diff'] = df.apply(
            lambda row: (row[f'{fighter}_career_stats_reversals'] - row[f'{fighter}_career_stats_reversals_against']) / np.sum(row[f'{fighter}_career_stats_time_fought']) * 100,
            axis=1
        )
    
        aux_dict[f'{fighter}_sub_acc'] = df.apply(
            lambda row: (row[f'{fighter}_career_stats_sub'] / row[f'{fighter}_career_stats_sub_attempts'] if row[f'{fighter}_career_stats_sub_attempts'] != 0 else 0.2) * 100,
            axis=1
        )
    
        aux_dict[f'{fighter}_sub_acc_against'] = df.apply(
            lambda row: (row[f'{fighter}_career_stats_subbed'] / row[f'{fighter}_career_stats_sub_attempts_against'] if row[f'{fighter}_career_stats_sub_attempts_against'] != 0 else 0.1) * 100,
            axis=1
        )
    
    new_df['Fight_TD_per_round_diff'] = aux_dict['Fighter_A_td_diff'] - aux_dict['Fighter_B_td_diff']
    new_df['Fight_control_pct_diff'] = aux_dict['Fighter_A_control_pct'] - aux_dict['Fighter_B_control_pct']
    # df.drop(['Fighter_A_strikes_head_diff', 'Fighter_B_strikes_head_diff', 'Fighter_A_strikes_body_diff', 'Fighter_B_strikes_body_diff'], axis=1)

    new_df['Fight_top_position_pct_diff'] = aux_dict['Fighter_A_top_pct'] - aux_dict['Fighter_B_top_pct']
    # df['Fighter_A_sub_acc_diff'] = df.apply(
    #     lambda row: row['Fighter_A_sub_acc'] - row['Fighter_A_sub_acc_against'],
    #     axis=1
    # )
    aux_dict['Fighter_A_sub_acc_diff'] = aux_dict['Fighter_A_sub_acc'] - aux_dict['Fighter_A_sub_acc_against']

    # aux_dict['Fighter_B_sub_acc_diff'] = df.apply(
    #     lambda row: row['Fighter_B_sub_acc'] - row['Fighter_B_sub_acc_against'],
    #     axis=1
    # )
    aux_dict['Fighter_B_sub_acc_diff'] = aux_dict['Fighter_B_sub_acc'] - aux_dict['Fighter_B_sub_acc_against']
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
        
        aux_dict[f'{fighter}_recovery_score'] = df[f'{fighter}_context_stats_recovery_time'].apply(recovery_score)
    
    
        # df[f'{fighter}_median_recovery_diff'] = df.apply(
        #     lambda row: np.median(row[f'{fighter}_career_stats_recovery_time']) - np.median(row[f'{fighter}_career_stats_opponent_stats_recovery_time']),
        #     axis=1
        # )
        aux_dict[f'{fighter}_median_recovery_score_diff'] = df.apply(
            lambda row: np.median([recovery_score(d) for d in row[f'{fighter}_career_stats_recovery_time']]) -
                        np.median([recovery_score(d) for d in row[f'{fighter}_career_stats_opponent_stats_recovery_time']]),
            axis=1
        )
    
        
        # df[f'{fighter}_mean_recovery_diff'] = df.apply(
        #     lambda row: np.mean(row[f'{fighter}_career_stats_recovery_time']) - np.mean(row[f'{fighter}_career_stats_opponent_stats_recovery_time']),
        #     axis=1
        # )
        aux_dict[f'{fighter}_mean_recovery_score_diff'] = df.apply(
            lambda row: np.mean([recovery_score(d) for d in row[f'{fighter}_career_stats_recovery_time']]) -
                        np.mean([recovery_score(d) for d in row[f'{fighter}_career_stats_opponent_stats_recovery_time']]),
            axis=1
        )
    
        
        # df[f'{fighter}_recovery_adv_pct'] = df.apply(
        #     lambda row: np.sum(np.array(row[f'{fighter}_career_stats_recovery_time']) > np.array(row[f'{fighter}_career_stats_opponent_stats_recovery_time'])) / 
        #                       np.array(row[f'{fighter}_career_stats_n_fights']),
        #     axis=1
        # )
        aux_dict[f'{fighter}_recovery_score_adv_pct'] = df.apply(
            lambda row: np.mean([
                recovery_score(f_r) > recovery_score(o_r)
                for f_r, o_r in zip(row[f'{fighter}_career_stats_recovery_time'],
                                    row[f'{fighter}_career_stats_opponent_stats_recovery_time'])
            ]),
            axis=1
        )

        aux_dict[f'{fighter}_avg_fight_time'] = df.apply(
            lambda row: weighted_average(row[f'{fighter}_career_stats_time_fought']),
            axis=1
        )
    
    
    new_df['Fight_avg_fight_time_diff'] = aux_dict['Fighter_B_avg_fight_time'] - aux_dict['Fighter_A_avg_fight_time']
    new_df['Fight_median_historic_recovery_diff'] = aux_dict['Fighter_B_median_recovery_score_diff'] - aux_dict['Fighter_A_median_recovery_score_diff']
    new_df['Fight_mean_historic_recovery_diff'] = aux_dict['Fighter_B_mean_recovery_score_diff'] - aux_dict['Fighter_A_mean_recovery_score_diff']
    
    
    
    # import numpy as np
    
    # age_idcs = np.where(np.array(df[f'{fighter}_career_stats_opponent_stats_age'])) > 0
    
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
    
        
        # df[f'{fighter}_age_adv_pct'] = df.apply(
        #     lambda row: np.sum(np.array(row[f'{fighter}_career_stats_age']) < np.array(row[f'{fighter}_career_stats_opponent_stats_age'])) / 
        #                       np.array(row[f'{fighter}_career_stats_n_fights']),
        #     axis=1
        # )
    
        # df[f'{fighter}_pct_old_opponents'] = df.apply(
        #     lambda row: (len(np.where(np.array(row[f'{fighter}_career_stats_opponent_stats_age']) > 33)[0]) + 
        #     2 * len(np.where(np.array(row[f'{fighter}_career_stats_opponent_stats_age']) > 35)[0]) + 
        #     3 * len(np.where(np.array(row[f'{fighter}_career_stats_opponent_stats_age']) > 37)[0]) + 
        #     4 * len(np.where(np.array(row[f'{fighter}_career_stats_opponent_stats_age']) > 39)[0])) 
        #     / np.array(row[f'{fighter}_career_stats_n_fights']),
        #     axis=1
        # )
        aux_dict[f'{fighter}_pct_old_opponents'] = df.apply(
            lambda row: weighted_old_opponent_score(row[f'{fighter}_career_stats_opponent_stats_age']),
            axis=1
        )
    
    
        # df[f'{fighter}_prime_age'] = df.apply(
        #     lambda row: 4 if (row[f'{fighter}_basic_stats_Age'] < 33 and row[f'{fighter}_basic_stats_Age'] > 24)
        #     else 3 if row[f'{fighter}_basic_stats_Age'] < 35
        #     else 2 if row[f'{fighter}_basic_stats_Age'] < 37
        #     else 1 if row[f'{fighter}_basic_stats_Age'] < 39
        #     else 0,
        #     axis=1
        # )
        aux_dict[f'{fighter}_prime_age'] = df[f'{fighter}_basic_stats_Age'].apply(continuous_prime_age)
    
    
    # here, lower value means fighter A had historically more advantage, so i've inversed the order of substraction
    new_df['Fight_median_historic_age_diff'] = aux_dict['Fighter_B_median_age_diff'] - aux_dict['Fighter_A_median_age_diff']
    new_df['Fight_mean_historic_age_diff'] = aux_dict['Fighter_B_mean_age_diff'] - aux_dict['Fighter_A_mean_age_diff']
    new_df['Fight_historic_age_adv_count_diff'] = aux_dict['Fighter_B_age_adv_pct'] - aux_dict['Fighter_A_age_adv_pct']
    new_df['Fight_historic_old_opponents_diff'] = aux_dict['Fighter_B_pct_old_opponents'] - aux_dict['Fighter_A_pct_old_opponents']
    
    # print(df['Fighter_A_age_adv_pct'])
    # print(df['Fighter_B_age_adv_pct'])
    
    
    aux_dict['Fighter_A_win_rate'] = df.apply(
        lambda row: (row['Fighter_A_career_stats_wins'] / row['Fighter_A_career_stats_n_fights'] if row['Fighter_A_career_stats_n_fights'] != 0 else 0.5) * 100,
        axis=1
    )
    
    aux_dict['Fighter_B_win_rate'] = df.apply(
        lambda row: (row['Fighter_B_career_stats_wins'] / row['Fighter_B_career_stats_n_fights'] if row['Fighter_B_career_stats_n_fights'] != 0 else 0.5) * 100,
        axis=1
    )

    finishing_rate, finished_rate, finishing_diff = {}, {}, {}
    
    for fighter in ["Fighter_A", "Fighter_B"]:
    
        # df[f'{fighter}_finishing_rate']
        finishing_rate[fighter] = df.apply(
            lambda row: ((row[f'{fighter}_career_stats_ko'] + row[f'{fighter}_career_stats_sub']) / row[f'{fighter}_career_stats_n_fights'] if row[f'{fighter}_career_stats_n_fights'] != 0 else 0.5) * 100,
            axis=1
        )
    
        # df[f'{fighter}_finished_rate'] = 
        finished_rate[fighter] = df.apply(
            lambda row: ((row[f'{fighter}_career_stats_kod'] + row[f'{fighter}_career_stats_subbed']) / row[f'{fighter}_career_stats_n_fights'] if row[f'{fighter}_career_stats_n_fights'] != 0 else 0.5) * 100,
            axis=1
        )
    
        # df[f'{fighter}_finishing_diff'] = 
        finishing_diff[fighter] = finishing_rate[fighter] - finished_rate[fighter]

    new_df['Fight_finishing_ovr_diff'] = finishing_rate['Fighter_A'] - finishing_rate['Fighter_B']
    new_df['Fight_finishing_rate_diff'] = finished_rate['Fighter_A'] - finished_rate['Fighter_B']
    new_df['Fight_finished_rate_diff'] = finishing_diff['Fighter_A'] - finishing_diff['Fighter_B']
        
    # new_columns = {}

    # for fighter in ["Fighter_A", "Fighter_B"]:
    #     finishing_rate = []
    #     finished_rate = []
    #     finishing_diff = []
    
    #     for _, row in df.iterrows():
    #         fights = row[f'{fighter}_career_stats_n_fights'] or 1  # avoid div by 0
    #         rate = ((row[f'{fighter}_career_stats_ko'] + row[f'{fighter}_career_stats_sub']) / fights) * 100
    #         rate_taken = ((row[f'{fighter}_career_stats_kod'] + row[f'{fighter}_career_stats_subbed']) / fights) * 100
    
    #         finishing_rate.append(rate)
    #         finished_rate.append(rate_taken)
    #         finishing_diff.append(rate - rate_taken)
    
    #     new_columns[f'{fighter}_finishing_rate'] = finishing_rate
    #     new_columns[f'{fighter}_finished_rate'] = finished_rate
    #     new_columns[f'{fighter}_finishing_diff'] = finishing_diff
    
    # # Add all at once
    # df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    new_df['Fight_WR_diff'] = aux_dict['Fighter_A_win_rate'] - aux_dict['Fighter_B_win_rate']
    # new_df['Fight_finishing_ovr_diff'] = df['Fighter_A_finishing_diff'] - df['Fighter_B_finishing_diff']
    # new_df['Fight_finishing_rate_diff'] = df['Fighter_A_finishing_rate'] - df['Fighter_B_finishing_rate']
    # new_df['Fight_finished_rate_diff'] = df['Fighter_B_finished_rate'] - df['Fighter_A_finished_rate']

    


    
    for fighter in ["Fighter_A", "Fighter_B"]:
    
        aux_dict[f'{fighter}_opponents_WR'] = df.apply(
            lambda row: np.sum(row[f'{fighter}_career_stats_opponent_stats_n_wins']) / np.sum(row[f'{fighter}_career_stats_opponent_stats_n_fights']),
            axis=1
        )

        aux_dict[f'{fighter}_pct_opponents_title_wins'] = df.apply( # championship level opposition? specially lately
            lambda row: weighted_average(row[f'{fighter}_career_stats_opponent_stats_n_title_wins']),# / row[f'{fighter}_career_stats_n_fights'], # i consider how great are opponents on average, that's why i divide by the same fighter's n fights
            axis=1
        )
        
    new_df['Fight_opponents_WR_diff'] = aux_dict['Fighter_A_opponents_WR'] - aux_dict['Fighter_B_opponents_WR']
    new_df['Fight_opponents_title_wins_diff'] = aux_dict['Fighter_A_pct_opponents_title_wins'] - aux_dict['Fighter_B_pct_opponents_title_wins']

    for fighter in ['Fighter_A', 'Fighter_B']:
        new_df[f'{fighter}_young'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] < 25 else 0,
            axis=1
        )
        new_df[f'{fighter}_prime_young'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] < 30 else 0,
            axis=1
        )
        new_df[f'{fighter}_prime'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] < 33 else 0,
            axis=1
        )
        new_df[f'{fighter}_prime_old'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] < 35 else 0,
            axis=1
        )
        new_df[f'{fighter}_declining'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] < 37 else 0,
            axis=1
        )
        new_df[f'{fighter}_veteran'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] < 39 else 0,
            axis=1
        )
        new_df[f'{fighter}_ancient'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] > 39 else 0,
            axis=1
        )

    for fighter in ['Fighter_A', 'Fighter_B']:
        new_df[f'{fighter}_young'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] < 25 else 0,
            axis=1
        )
        new_df[f'{fighter}_prime_young'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] in (25, 30) else 0,
            axis=1
        )
        new_df[f'{fighter}_prime'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] in (30, 33) else 0,
            axis=1
        )
        new_df[f'{fighter}_prime_old'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] in (33, 35) else 0,
            axis=1
        )
        new_df[f'{fighter}_declining'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] in (35, 37) else 0,
            axis=1
        )
        new_df[f'{fighter}_veteran'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] in (37, 38.5) else 0,
            axis=1
        )
        new_df[f'{fighter}_ancient'] = df.apply(
            lambda row: 1 if row[f'{fighter}_basic_stats_Age'] > 38.5 else 0,
            axis=1
        )

    for fighter in ['Fighter_A', 'Fighter_B']:
        new_df[f'{fighter}_very_short_recovery'] = df.apply(
            lambda row: 1 if row[f'{fighter}_context_stats_recovery_time'] in (1, 30) else 0,
            axis=1
        )
        new_df[f'{fighter}_short_recovery'] = df.apply(
            lambda row: 1 if row[f'{fighter}_context_stats_recovery_time'] in (30, 90) else 0,
            axis=1
        )
        new_df[f'{fighter}_moderate_recovery'] = df.apply(
            lambda row: 1 if row[f'{fighter}_context_stats_recovery_time'] in (90, 150) else 0,
            axis=1
        )
        new_df[f'{fighter}_long_recovery'] = df.apply(
            lambda row: 1 if row[f'{fighter}_context_stats_recovery_time'] in (150, 250) else 0,
            axis=1
        )
        new_df[f'{fighter}_very_long_recovery'] = df.apply(
            lambda row: 1 if row[f'{fighter}_context_stats_recovery_time'] in (250, 375) else 0,
            axis=1
        )
        new_df[f'{fighter}_ring_rust'] = df.apply(
            lambda row: 1 if row[f'{fighter}_context_stats_recovery_time'] > 375 else 0,
            axis=1
        )
    
    new_df['Fight_age_diff'] = df['Fighter_B_basic_stats_Age'] - df['Fighter_A_basic_stats_Age']
    new_df['Fight_undefeated_diff'] = df['Fighter_A_career_stats_is_undefeated'].astype(int) - df['Fighter_B_career_stats_is_undefeated'].astype(int)
    # new_df['Fight_recovery_diff'] = df['Fighter_A_recovery_score'] - df['Fighter_B_recovery_score']
    # new_df['Fight_prime_age_diff'] = df['Fighter_A_prime_age'] - df['Fighter_B_prime_age']
    new_df['Fight_reach_diff'] = df['Fighter_A_basic_stats_Reach'] - df['Fighter_B_basic_stats_Reach']
    new_df['Fight_height_diff'] = df['Fighter_A_basic_stats_Height'] - df['Fighter_B_basic_stats_Height']
    new_df['Fight_stance_0_diff'] = df['Fighter_A_basic_stats_Stance_0'] - df['Fighter_B_basic_stats_Stance_0']
    new_df['Fight_stance_1_diff'] = df['Fighter_A_basic_stats_Stance_1'] - df['Fighter_B_basic_stats_Stance_1']
    new_df['Fight_streak_diff'] = df['Fighter_A_context_stats_unbeaten_streak'] - df['Fighter_B_context_stats_unbeaten_streak']

    if full_df is None: # only feature that needs to travel through the dataset
        df['Fight_weight_diff'] = get_fights_weight_diff(df)
    else: # what exactly is this?
        df['Fight_weight_diff'] = [get_fights_weight_diff(df, full_df)]

    new_df['Fight_weight_diff'] = df['Fight_weight_diff']

    new_df['Fight_025_div_change'] = df.apply(
        lambda row: 1 if row['Fight_weight_diff'] in (0, 0.25) else 
        -1 if row['Fight_weight_diff'] in (-0.25, 0) else 0,
        axis=1
    )
    new_df['Fight_05_div_change'] = df.apply(
        lambda row: 1 if row['Fight_weight_diff'] in (0.25, 0.5) else 
        -1 if row['Fight_weight_diff'] in (-0.5, -0.25) else 0,
        axis=1
    )
    new_df['Fight_075_div_change'] = df.apply(
        lambda row: 1 if row['Fight_weight_diff'] in (0.5, 0.75) else 
        -1 if row['Fight_weight_diff'] in (-0.75, -0.5) else 0,
        axis=1
    )
    new_df['Fight_01_div_change'] = df.apply(
        lambda row: 1 if row['Fight_weight_diff'] in (0.75, 1) else 
        -1 if row['Fight_weight_diff'] in (-1, -0.75) else 0,
        axis=1
    )
    new_df['Fight_10_div_change'] = df.apply(
        lambda row: np.exp(row['Fight_weight_diff']) if row['Fight_weight_diff'] > 1 else 
        -np.exp(row['Fight_weight_diff']) if row['Fight_weight_diff'] < -1 else 0,
        axis=1
    )

    df.drop(['Fight_weight_diff'], axis=1)

    
    for fighter in ["Fighter_A", "Fighter_B"]:

        # df[f'{fighter}_moving_up'] = df.apply(is_moving_up(fighter), axis=1)

        if full_df is None:
            aux_dict[f'{fighter}_moving_up'] = df.apply(
                lambda row: is_moving_up(row, df, fighter),
                axis=1
            )
            
            aux_dict[f'{fighter}_moving_down'] = df.apply(
                lambda row: is_moving_down(row, df, fighter),
                axis=1
            )
        else:
            aux_dict[f'{fighter}_moving_up'] = df.apply(
                lambda row: is_moving_up(df.iloc[0], full_df, fighter),
                axis=1
            )
            
            aux_dict[f'{fighter}_moving_down'] = df.apply(
                lambda row: is_moving_down(df.iloc[0], full_df, fighter),
                axis=1
            )

    new_df['Fight_moving_up_diff'] = aux_dict['Fighter_A_moving_up'] - aux_dict['Fighter_B_moving_up'] # oversimplification? no anynmore?
    new_df['Fight_moving_down_diff'] = aux_dict['Fighter_A_moving_down'] - aux_dict['Fighter_B_moving_down']

        
    # mask = (df["Fighter_A_id"] == 689) | (df["Fighter_B_id"] == 689)
    # average_division_fighter_A = df.loc[mask, "Fight_info_division"].mean()
    # print("Average Fight_info_division:", average_division)

    # mask = (df["Fighter_A_id"] == 689) | (df["Fighter_B_id"] == 689)
    # average_division_fighter_B = df.loc[mask, "Fight_info_division"].mean()
    # print("Average Fight_info_division:", average_division)

    # new_df['Fight_weight_diff'] = average_division_fighter_A - average_division_fighter_B


    
    new_df = pd.DataFrame.from_dict(new_df)
    return new_df