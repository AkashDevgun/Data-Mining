import numpy as np
import pandas as pd
import datetime

# loading dataset
fulldatadf = pd.read_csv('AllSeasonsData.csv')

print fulldatadf.columns

# change data types of Dates
fulldatadf['date'] = pd.to_datetime(fulldatadf['date'].astype(str), format = '%Y%m%d')


fulldatadf = fulldatadf[fulldatadf.playoffs == 0]

# create an empty list for all features
features = []

# Through all the seasons to iterate
allseasons = fulldatadf.season.unique()

for eachseason in allseasons:
	# DataFrame of one season
	df_eachseason = fulldatadf[fulldatadf.season == eachseason]

	# get a unique list of the days to iterate through
	days = df_eachseason.date.unique()

	# iterate through the days
	for eachday in days:
		print eachday
		
		#DataFrame At the day and DataFrame before that day
		df_attheday = df_eachseason[df_eachseason.date == eachday]
		df_beforetheday = df_eachseason[df_eachseason.date < eachday]

		#Form each Feature List
		for index, rowvalue in df_attheday.iterrows():
			
			eachFeaturelist = []
			colnames = []

			# Variables which will be as it is
			for each in ['date',
					  'line',
					  'team', 'o:team']:
				eachFeaturelist.append(rowvalue[each])
			colnames.append('date')
			colnames.append('line')
			colnames.append('home_team')
			colnames.append('away_team')

			# Only for Western conferences
			eachFeaturelist.append((rowvalue['conference'] == 'Western') * 1)
			eachFeaturelist.append((rowvalue['o:conference'] == 'Western') * 1)
			colnames.append('home_conference')
			colnames.append('away_conference')

			#Mean Values for season to date,l10, l5 and l3 data frames
			df_mean_home = df_beforetheday[df_beforetheday['team'] == rowvalue['team']].mean()
			df_mean_away = df_beforetheday[df_beforetheday['team'] == rowvalue['o:team']].mean()
			df_l10_home_mean = df_beforetheday[df_beforetheday['team'] == rowvalue['team']].tail(10).mean()
			df_l10_away_mean = df_beforetheday[df_beforetheday['team'] == rowvalue['o:team']].tail(10).mean()
			df_l5_home_mean = df_beforetheday[df_beforetheday['team'] == rowvalue['team']].tail(5).mean()
			df_l5_away_mean = df_beforetheday[df_beforetheday['team'] == rowvalue['o:team']].tail(5).mean()
			df_l3_home_mean = df_beforetheday[df_beforetheday['team'] == rowvalue['team']].tail(3).mean()
			df_l3_away_mean = df_beforetheday[df_beforetheday['team'] == rowvalue['o:team']].tail(3).mean()



			if rowvalue['ats margin'] > 0:
				parameter = 0
			else:
				parameter = 1

			eachFeaturelist.append(parameter)
			colnames.append('home_win')

			# assist to turnover ratio
			eachFeaturelist.append((df_mean_home['assists'] / df_mean_home['turnovers']) \
											/ (df_mean_away['assists'] / df_mean_away['turnovers']))
			colnames.append("ratio_assist2turnover")

			eachFeaturelist.append(df_mean_home['assists'] / df_mean_away['assists'])
			colnames.append("ratio_assists_season")

			eachFeaturelist.append(df_l10_home_mean['assists'] / df_l10_away_mean['assists'])
			colnames.append("ratio_assists_l10")

			eachFeaturelist.append(df_l5_home_mean['assists'] / df_l5_away_mean['assists'])
			colnames.append("ratio_assists_l5")

			eachFeaturelist.append(df_l3_home_mean['assists'] / df_l3_away_mean['assists'])
			colnames.append("ratio_assists_l3")


			eachFeaturelist.append((df_mean_home['assists'] / df_mean_away['assists'])/ (df_l10_home_mean['assists'] / df_l10_away_mean['assists']))
			colnames.append("ratio_assists_season_l10")


			eachFeaturelist.append((df_mean_home['assists'] / df_mean_away['assists'])/ (df_l5_home_mean['assists'] / df_l5_away_mean['assists']))
			colnames.append("ratio_assists_season_l5")

			eachFeaturelist.append((df_mean_home['assists'] / df_mean_away['assists'])/ (df_l3_home_mean['assists'] / df_l3_away_mean['assists']))
			colnames.append("ratio_assists_season_l3")

			eachFeaturelist.append(df_mean_home['line'] / df_mean_away['line'])
			colnames.append("ratio_spread_season")

			eachFeaturelist.append(df_l10_home_mean['line'] / df_l10_away_mean['line'])
			colnames.append("ratio_spread_l10")

			eachFeaturelist.append(df_l5_home_mean['line'] / df_l5_away_mean['line'])
			colnames.append("ratio_spread_l5")

			eachFeaturelist.append(df_l3_home_mean['line'] / df_l3_away_mean['line'])
			colnames.append("ratio_spread_l3")


			eachFeaturelist.append((df_mean_home['steals'] / df_mean_home['blocks']) \
											/ (df_mean_away['steals'] / df_mean_away['blocks']))
			colnames.append("ratio_steal2block")

			eachFeaturelist.append((df_l10_home_mean['steals'] / df_l10_home_mean['blocks']) \
											/ (df_l10_away_mean['steals'] / df_l10_away_mean['blocks']))
			colnames.append("ratio_steal2blockl10")


			eachFeaturelist.append((df_l5_home_mean['steals'] / df_l5_home_mean['blocks']) \
											/ (df_l5_away_mean['steals'] / df_l5_away_mean['blocks']))
			colnames.append("ratio_steal2blockl5")

			eachFeaturelist.append((df_l3_home_mean['steals'] / df_l3_home_mean['blocks']) \
											/ (df_l3_away_mean['steals'] / df_l3_away_mean['blocks']))
			colnames.append("ratio_steal2blockl3")

			eachFeaturelist.append(((df_mean_home['steals'] / df_mean_home['blocks']) \
											/ (df_mean_away['steals'] / df_mean_away['blocks'])) \
											/ ((df_l3_home_mean['steals'] / df_l3_home_mean['blocks']) \
											/ (df_l3_away_mean['steals'] / df_l3_away_mean['blocks'])) )
			colnames.append("steals_to_ratio_blocks_season_l3")


			eachFeaturelist.append((df_mean_home['margin at the half'] / df_mean_home['margin']) \
											/ (df_mean_away['margin at the half'] / df_mean_away['margin']))
			colnames.append("ratio_marginHalfvsFull_season")

			

			eachFeaturelist.append((df_l10_home_mean['margin at the half'] / df_l10_home_mean['margin']) \
											/ (df_l10_away_mean['margin at the half'] / df_l10_away_mean['margin']))
			colnames.append("ratio_marginHalfvsFull_l10")

			eachFeaturelist.append((df_l5_home_mean['margin at the half'] / df_l5_home_mean['margin']) \
											/ (df_l5_away_mean['margin at the half'] / df_l5_away_mean['margin']))
			colnames.append("ratio_marginHalfvsFull_l5")

			eachFeaturelist.append((df_l3_home_mean['margin at the half'] / df_l3_home_mean['margin']) \
												/ (df_l3_away_mean['margin at the half'] / df_l3_away_mean['margin']))
			colnames.append("ratio_marginHalfvsFull_l3")





			# three point fg % ratio
			eachFeaturelist.append((df_mean_home['three pointers made'] / df_mean_home['three pointers attempted']) \
											/ (df_mean_away['three pointers made'] / df_mean_away['three pointers attempted']))
			colnames.append("ratio_3pointfg_season")

			

			eachFeaturelist.append((df_l10_home_mean['three pointers made'] / df_l10_home_mean['three pointers attempted']) \
											/ (df_l10_away_mean['three pointers made'] / df_l10_away_mean['three pointers attempted']))
			colnames.append("ratio_3pointfg_l10")

			eachFeaturelist.append((df_l5_home_mean['three pointers made'] / df_l5_home_mean['three pointers attempted']) \
											/ (df_l5_away_mean['three pointers made'] / df_l5_away_mean['three pointers attempted']))
			colnames.append("ratio_3pointfg_l5")

			eachFeaturelist.append((df_l3_home_mean['three pointers made'] / df_l3_home_mean['three pointers attempted']) \
												/ (df_l3_away_mean['three pointers made'] / df_l3_away_mean['three pointers attempted']))
			colnames.append("ratio_3pointfg_l3")	


			eachFeaturelist.append(((df_mean_home['three pointers made'] / df_mean_home['three pointers attempted']) \
											/ (df_mean_away['three pointers made'] / df_mean_away['three pointers attempted'])) \
											/	(df_l10_home_mean['three pointers made'] / df_l10_home_mean['three pointers attempted']) \
											/ (df_l10_away_mean['three pointers made'] / df_l10_away_mean['three pointers attempted']))

			eachFeaturelist.append(((df_mean_home['three pointers made'] / df_mean_home['three pointers attempted']) \
											/ (df_mean_away['three pointers made'] / df_mean_away['three pointers attempted'])) \
											/	(df_l5_home_mean['three pointers made'] / df_l5_home_mean['three pointers attempted']) \
											/ (df_l5_away_mean['three pointers made'] / df_l5_away_mean['three pointers attempted']))


			eachFeaturelist.append(((df_mean_home['three pointers made'] / df_mean_home['three pointers attempted']) \
											/ (df_mean_away['three pointers made'] / df_mean_away['three pointers attempted'])) \
											/	(df_l3_home_mean['three pointers made'] / df_l3_home_mean['three pointers attempted']) \
											/ (df_l3_away_mean['three pointers made'] / df_l3_away_mean['three pointers attempted']))


			colnames.append("ratio_3pointfg_season_l10")

			colnames.append("ratio_3pointfg_season_l5")

			colnames.append("ratio_3pointfg_season_l3")		


			# three pointers attempted ratio
			eachFeaturelist.append(df_mean_home['three pointers made'] / df_mean_away['three pointers attempted'])
			colnames.append("ratio_3pointatt")		


			# free throw % ratio
			eachFeaturelist.append((df_mean_home['free throws made'] / df_mean_home['free throws attempted']) \
											/ (df_mean_away['free throws made'] / df_mean_away['free throws attempted']))
			colnames.append("ratio_freethrowfg")

			eachFeaturelist.append((df_l10_home_mean['free throws made'] / df_l10_home_mean['free throws attempted']) \
											/ (df_l10_away_mean['free throws made'] / df_l10_away_mean['free throws attempted']))
			colnames.append("ratio_freethrowfg_l10")

			eachFeaturelist.append((df_l5_home_mean['free throws made'] / df_l5_home_mean['free throws attempted']) \
											/ (df_l5_away_mean['free throws made'] / df_l5_away_mean['free throws attempted']))
			colnames.append("ratio_freethrowfg_l5")

			eachFeaturelist.append((df_l3_home_mean['free throws made'] / df_l3_home_mean['free throws attempted']) \
											/ (df_l3_away_mean['free throws made'] / df_l3_away_mean['free throws attempted']))
			colnames.append("ratio_freethrowfg_l3")



			eachFeaturelist.append((df_mean_home['field goals made'] / df_mean_home['field goals attempted']) \
											/ (df_mean_away['field goals made'] / df_mean_away['field goals attempted']))
			colnames.append("ratio_fg")

			eachFeaturelist.append((df_l10_home_mean['field goals made'] / df_l10_home_mean['field goals attempted']) \
											/ (df_l10_away_mean['field goals made'] / df_l10_away_mean['field goals attempted']))
			colnames.append("ratio_fg_l10")

			eachFeaturelist.append((df_l5_home_mean['field goals made'] / df_l5_home_mean['field goals attempted']) \
											/ (df_l5_away_mean['field goals made'] / df_l5_away_mean['field goals attempted']))
			colnames.append("ratio_fg_l5")

			eachFeaturelist.append((df_l3_home_mean['field goals made'] / df_l3_home_mean['field goals attempted']) \
											/ (df_l3_away_mean['field goals made'] / df_l3_away_mean['field goals attempted']))
			colnames.append("ratio_fg_l3")

			eachFeaturelist.append(((df_mean_home['field goals made'] / df_mean_home['field goals attempted']) \
											/ (df_mean_away['field goals made'] / df_mean_away['field goals attempted']))/ ((df_l10_home_mean['field goals made'] / df_l10_home_mean['field goals attempted']) \
											/ (df_l10_away_mean['field goals made'] / df_l10_away_mean['field goals attempted'])))

			colnames.append("fgpct_season_vs_l10_ratio")

			eachFeaturelist.append(((df_l10_home_mean['field goals made'] / df_l10_home_mean['field goals attempted']) \
											/ (df_l10_away_mean['field goals made'] / df_l10_away_mean['field goals attempted']))/ ((df_l3_home_mean['field goals made'] / df_l3_home_mean['field goals attempted']) \
											/ (df_l3_away_mean['field goals made'] / df_l3_away_mean['field goals attempted'])))

			colnames.append("fgpct_l10_vs_l3_ratio")


			# three pointers attempted ratio
			eachFeaturelist.append(df_mean_home['free throws made'] / df_mean_away['free throws attempted'])
			colnames.append("ratio_freethrowatt")	


			# ratio assists
			eachFeaturelist.append(df_mean_home['assists'] / df_mean_away['assists'])
			colnames.append("ratio_assists")


			# points in the paint ratio
			eachFeaturelist.append(df_mean_home['points in the paint'] / df_mean_away['points in the paint'])
			colnames.append("ratio_pointsinthepaint")

			eachFeaturelist.append(df_l3_home_mean['points in the paint'] / df_l3_away_mean['points in the paint'])
			colnames.append("ratio_pointsinthepaint_l3")

			eachFeaturelist.append(df_l5_home_mean['points in the paint'] / df_l5_away_mean['points in the paint'])
			colnames.append("ratio_pointsinthepaint_l5")

			eachFeaturelist.append(df_l10_home_mean['points in the paint'] / df_l10_away_mean['points in the paint'])
			colnames.append("ratio_pointsinthepaint_l10")


			# turnovers ratio
			eachFeaturelist.append(df_mean_home['turnovers'] / df_mean_away['turnovers'])
			colnames.append("ratio_turnovers_season")

			eachFeaturelist.append(df_l10_home_mean['turnovers'] / df_l10_away_mean['turnovers'])
			colnames.append("ratio_turnovers_l10")

			eachFeaturelist.append(df_l5_home_mean['turnovers'] / df_l5_away_mean['turnovers'])
			colnames.append("ratio_turnovers_l5")

			eachFeaturelist.append(df_l3_home_mean['turnovers'] / df_l3_away_mean['turnovers'])
			colnames.append("ratio_turnovers_l3")

			eachFeaturelist.append((df_mean_home['turnovers'] / df_mean_away['turnovers'])/(df_l10_home_mean['turnovers'] / df_l10_away_mean['turnovers']))
			colnames.append("ratio_turnovers_season_l10")

			eachFeaturelist.append((df_mean_home['turnovers'] / df_mean_away['turnovers'])/(df_l5_home_mean['turnovers'] / df_l5_away_mean['turnovers']))
			colnames.append("ratio_turnovers_season_l5")

			eachFeaturelist.append((df_mean_home['turnovers'] / df_mean_away['turnovers'])/(df_l3_home_mean['turnovers'] / df_l3_away_mean['turnovers']))
			colnames.append("ratio_turnovers_season_l3")




			# offensive rebound ratio
			eachFeaturelist.append(df_mean_home['offensive rebounds'] / df_mean_away['offensive rebounds'])
			colnames.append("ratio_orebounds_season")

			eachFeaturelist.append(df_l10_home_mean['offensive rebounds'] / df_l10_away_mean['offensive rebounds'])
			colnames.append("ratio_orebounds_l10")

			eachFeaturelist.append(df_l5_home_mean['offensive rebounds'] / df_l5_away_mean['offensive rebounds'])
			colnames.append("ratio_orebounds_l5")

			eachFeaturelist.append(df_l3_home_mean['offensive rebounds'] / df_l3_away_mean['offensive rebounds'])
			colnames.append("ratio_orebounds_l3")			


			# rest difference
			eachFeaturelist.append(df_mean_home['rest'] - df_mean_away['rest'])
			colnames.append("difference_rest_season")

			eachFeaturelist.append(df_l10_home_mean['rest'] - df_l10_away_mean['rest'])
			colnames.append("difference_rest_l10")

			eachFeaturelist.append(df_l5_home_mean['rest'] - df_l5_away_mean['rest'])
			colnames.append("difference_rest_l5")

			eachFeaturelist.append(df_l3_home_mean['rest'] - df_l3_away_mean['rest'])
			colnames.append("difference_rest_l3")				


			# ats streak difference
			eachFeaturelist.append(df_mean_home['ats streak'] - df_mean_away['ats streak'])
			colnames.append("difference_atsstreak_season")

			eachFeaturelist.append(df_l10_home_mean['ats streak'] - df_l10_away_mean['ats streak'])
			colnames.append("difference_atsstreak_l10")

			eachFeaturelist.append(df_l5_home_mean['ats streak'] - df_l5_away_mean['ats streak'])
			colnames.append("difference_atsstreak_l5")

			eachFeaturelist.append(df_l3_home_mean['ats streak'] - df_l3_away_mean['ats streak'])
			colnames.append("difference_atsstreak_l3")	


			# ats margin ratio
			eachFeaturelist.append(df_mean_home['ats margin'] / df_mean_away['ats margin'])
			colnames.append("ratio_atsmargin_season")

			eachFeaturelist.append(df_l10_home_mean['ats margin'] / df_l10_away_mean['ats margin'])
			colnames.append("ratio_atsmargin_l10")

			eachFeaturelist.append(df_l5_home_mean['ats margin'] / df_l5_away_mean['ats margin'])
			colnames.append("ratio_atsmargin_l5")

			eachFeaturelist.append(df_l3_home_mean['ats margin'] / df_l3_away_mean['ats margin'])
			colnames.append("ratio_atsmargin_l3")

			eachFeaturelist.append((df_mean_home['ats margin'] / df_mean_away['ats margin'])/(df_l10_home_mean['ats margin'] / df_l10_away_mean['ats margin']))
			colnames.append("ratio_atsmargin_season_l10")

			eachFeaturelist.append((df_mean_home['ats margin'] / df_mean_away['ats margin'])/(df_l5_home_mean['ats margin'] / df_l5_away_mean['ats margin']))
			colnames.append("ratio_atsmargin_season_l5")

			eachFeaturelist.append((df_mean_home['ats margin'] / df_mean_away['ats margin'])/(df_l3_home_mean['ats margin'] / df_l3_away_mean['ats margin']))
			colnames.append("ratio_atsmargin_season_l3")


			# blocks ratio
			eachFeaturelist.append(df_mean_home['blocks'] / df_mean_away['blocks'])
			colnames.append("ratio_blocks_season")

			eachFeaturelist.append(df_l10_home_mean['blocks'] / df_l10_away_mean['blocks'])
			colnames.append("ratio_blocks_l10")

			eachFeaturelist.append(df_l5_home_mean['blocks'] / df_l5_away_mean['blocks'])
			colnames.append("ratio_blocks_l5")

			eachFeaturelist.append(df_l5_home_mean['blocks'] / df_l3_away_mean['blocks'])
			colnames.append("ratio_blocks_l3")

			eachFeaturelist.append(rowvalue['line'])
			colnames.append("spread")


			# points ratio
			eachFeaturelist.append(df_mean_home['points'] / df_mean_away['points'])
			colnames.append("ratio_points")

			# winning % ratio
			eachFeaturelist.append((df_mean_home['wins'] / df_mean_home['losses']) \
											/ (df_mean_away['wins'] / df_mean_away['losses']))
			colnames.append("ratio_winperc")

			# winning % ratio - last 10
			eachFeaturelist.append((df_l10_home_mean['wins'] / df_l10_home_mean['losses']) \
											/ (df_l10_away_mean['wins'] / df_l10_away_mean['losses']))
			colnames.append("ratio_winperc_l10")


			eachFeaturelist.append(df_mean_home['steals'] / df_mean_away['steals'])
			colnames.append("ratio_steals_season")

			eachFeaturelist.append(df_l10_home_mean['steals'] / df_l10_away_mean['steals'])
			colnames.append("ratio_steals_l10")

			eachFeaturelist.append(df_l5_home_mean['steals'] / df_l5_away_mean['steals'])
			colnames.append("ratio_steals_l5")

			eachFeaturelist.append(df_l3_home_mean['steals'] / df_l3_away_mean['steals'])
			colnames.append("ratio_steals_l3")


			eachFeaturelist.append((df_mean_home['steals'] / df_mean_away['steals'])/(df_l10_home_mean['steals'] / df_l10_away_mean['steals']))
			colnames.append("ratio_steals_season_l10")





			eachFeaturelist.append(df_mean_home['margin'] / df_mean_away['margin'])
			colnames.append("ratio_margin_season")

			eachFeaturelist.append(df_l10_home_mean['margin'] / df_l10_away_mean['margin'])
			colnames.append("ratio_margin_l10")

			eachFeaturelist.append(df_l5_home_mean['margin'] / df_l5_away_mean['margin'])
			colnames.append("ratio_margin_l5")

			eachFeaturelist.append(df_l3_home_mean['margin'] / df_l3_away_mean['margin'])
			colnames.append("ratio_margin_l3")

			eachFeaturelist.append((df_mean_home['margin'] / df_mean_away['margin'])/(df_l3_home_mean['margin'] / df_l3_away_mean['margin']))
			colnames.append("ratio_margin_season_l3")

			eachFeaturelist.append((df_mean_home['margin'] / df_mean_away['margin'])/(df_l10_home_mean['margin'] / df_l10_away_mean['margin']))
			colnames.append("ratio_margin_season_l10")

			eachFeaturelist.append((df_mean_home['margin'] / df_mean_away['margin'])/(df_l5_home_mean['margin'] / df_l5_away_mean['margin']))
			colnames.append("ratio_margin_season_l5")

			eachFeaturelist.append((df_l10_home_mean['margin'] / df_l10_away_mean['margin'])/(df_l3_home_mean['margin'] / df_l3_away_mean['margin']))
			colnames.append("ratio_margin_l10_l3")





			eachFeaturelist.append(df_mean_home['team rebounds'] / df_mean_away['team rebounds'])
			colnames.append("ratio_teamrebounds_season")

			eachFeaturelist.append(df_l10_home_mean['team rebounds'] / df_l10_away_mean['team rebounds'])
			colnames.append("ratio_teamrebounds_l10")

			eachFeaturelist.append(df_l5_home_mean['team rebounds'] / df_l5_away_mean['team rebounds'])
			colnames.append("ratio_teamrebounds_l5")

			eachFeaturelist.append(df_l3_home_mean['team rebounds'] / df_l3_away_mean['team rebounds'])
			colnames.append("ratio_teamrebounds_l3")

			eachFeaturelist.append((df_mean_home['team rebounds'] / df_mean_away['team rebounds'])/(df_l10_home_mean['team rebounds'] / df_l10_away_mean['team rebounds']))
			colnames.append("ratio_teamrebounds_season_l10")

			eachFeaturelist.append((df_mean_home['team rebounds'] / df_mean_away['team rebounds'])/(df_l5_home_mean['team rebounds'] / df_l5_away_mean['team rebounds']))
			colnames.append("ratio_teamrebounds_season_l5")

			eachFeaturelist.append((df_mean_home['team rebounds'] / df_mean_away['team rebounds'])/(df_l3_home_mean['team rebounds'] / df_l3_away_mean['team rebounds']))
			colnames.append("ratio_teamrebounds_season_l3")




			eachFeaturelist.append(df_mean_home['biggest lead'] / df_mean_away['biggest lead'])
			colnames.append("ratio_biggestlead_season")

			eachFeaturelist.append(df_l10_home_mean['biggest lead'] / df_l10_away_mean['biggest lead'])
			colnames.append("ratio_biggestlead_l10")

			eachFeaturelist.append(df_l5_home_mean['biggest lead'] / df_l5_away_mean['biggest lead'])
			colnames.append("ratio_biggestlead_l5")

			eachFeaturelist.append(df_l3_home_mean['biggest lead'] / df_l3_away_mean['biggest lead'])
			colnames.append("ratio_biggestlead_l3")

			eachFeaturelist.append((df_mean_home['biggest lead'] / df_mean_away['biggest lead'])/ (df_l10_home_mean['biggest lead'] / df_l10_away_mean['biggest lead']))
			colnames.append("ratio_biggestlead_season_l10")

			eachFeaturelist.append((df_mean_home['biggest lead'] / df_mean_away['biggest lead'])/ (df_l5_home_mean['biggest lead'] / df_l5_away_mean['biggest lead']))
			colnames.append("ratio_biggestlead_season_l5")

			eachFeaturelist.append((df_mean_home['biggest lead'] / df_mean_away['biggest lead'])/ (df_l3_home_mean['biggest lead'] / df_l3_away_mean['biggest lead']))
			colnames.append("ratio_biggestlead_season_l3")



			eachFeaturelist.append(df_mean_home['defensive rebounds'] / df_mean_away['defensive rebounds'])
			colnames.append("ratio_defensiverebounds_season")

			eachFeaturelist.append(df_l10_home_mean['defensive rebounds'] / df_l10_away_mean['defensive rebounds'])
			colnames.append("ratio_defensiverebounds_l10")

			eachFeaturelist.append(df_l5_home_mean['defensive rebounds'] / df_l5_away_mean['defensive rebounds'])
			colnames.append("ratio_defensiverebounds_l5")

			eachFeaturelist.append(df_l3_home_mean['defensive rebounds'] / df_l3_away_mean['defensive rebounds'])
			colnames.append("ratio_defensiverebounds_l3")

			eachFeaturelist.append((df_mean_home['defensive rebounds'] / df_mean_away['defensive rebounds'])/ (df_l10_home_mean['defensive rebounds'] / df_l10_away_mean['defensive rebounds']))
			colnames.append("ratio_defensiverebounds_season_l10")

			eachFeaturelist.append((df_mean_home['defensive rebounds'] / df_mean_away['defensive rebounds'])/ (df_l5_home_mean['defensive rebounds'] / df_l5_away_mean['defensive rebounds']))
			colnames.append("ratio_defensiverebounds_season_l5")

			eachFeaturelist.append((df_mean_home['defensive rebounds'] / df_mean_away['defensive rebounds'])/ (df_l3_home_mean['defensive rebounds'] / df_l3_away_mean['defensive rebounds']))
			colnames.append("ratio_defensiverebounds_season_l3")




			eachFeaturelist.append(df_mean_home['fouls'] / df_mean_away['fouls'])
			colnames.append("ratio_fouls_season")

			eachFeaturelist.append(df_l10_home_mean['fouls'] / df_l10_away_mean['fouls'])
			colnames.append("ratio_fouls_l10")

			eachFeaturelist.append(df_l5_home_mean['fouls'] / df_l5_away_mean['fouls'])
			colnames.append("ratio_fouls_l5")

			eachFeaturelist.append(df_l3_home_mean['fouls'] / df_l3_away_mean['fouls'])
			colnames.append("ratio_fouls_l3")

			eachFeaturelist.append((df_mean_home['fouls'] / df_mean_away['fouls'])/(df_l10_home_mean['fouls'] / df_l10_away_mean['fouls']))
			colnames.append("ratio_fouls_season_l10")

			eachFeaturelist.append((df_mean_home['fouls'] / df_mean_away['fouls'])/(df_l5_home_mean['fouls'] / df_l5_away_mean['fouls']))
			colnames.append("ratio_fouls_season_l5")

			eachFeaturelist.append((df_mean_home['fouls'] / df_mean_away['fouls'])/(df_l3_home_mean['fouls'] / df_l3_away_mean['fouls']))
			colnames.append("ratio_fouls_season_l3")

			eachFeaturelist.append((df_l10_home_mean['fouls'] / df_l10_away_mean['fouls'])/(df_l3_home_mean['fouls'] / df_l3_away_mean['fouls']))
			colnames.append("ratio_fouls_l10_l3")







			eachFeaturelist.append(df_mean_home['fast break points'] / df_mean_away['fast break points'])
			colnames.append("ratio_fastbreakpoints_season")

			eachFeaturelist.append(df_l10_home_mean['fast break points'] / df_l10_away_mean['fast break points'])
			colnames.append("ratio_fastbreakpoints_l10")

			eachFeaturelist.append(df_l5_home_mean['fast break points'] / df_l5_away_mean['fast break points'])
			colnames.append("ratio_fastbreakpoints_l5")

			eachFeaturelist.append(df_l3_home_mean['fast break points'] / df_l3_away_mean['fast break points'])
			colnames.append("ratio_fastbreakpoints_l3")


			eachFeaturelist.append((df_mean_home['fast break points'] / df_mean_away['fast break points'])/(df_l10_home_mean['fast break points'] / df_l10_away_mean['fast break points']))
			colnames.append("ratio_fastbreakpoints_season_l10")

			eachFeaturelist.append((df_mean_home['fast break points'] / df_mean_away['fast break points'])/(df_l5_home_mean['fast break points'] / df_l5_away_mean['fast break points']))
			colnames.append("ratio_fastbreakpoints_season_l5")

			eachFeaturelist.append((df_mean_home['fast break points'] / df_mean_away['fast break points'])/(df_l3_home_mean['fast break points'] / df_l3_away_mean['fast break points']))
			colnames.append("ratio_fastbreakpoints_season_l3")






			eachFeaturelist.append(df_mean_home['streak'] / df_mean_away['streak'])
			colnames.append("ratio_streak_season")

			eachFeaturelist.append(df_l10_home_mean['streak'] / df_l10_away_mean['streak'])
			colnames.append("ratio_streak_l10")

			eachFeaturelist.append(df_l5_home_mean['streak'] / df_l5_away_mean['streak'])
			colnames.append("ratio_streak_l5")

			eachFeaturelist.append(df_l3_home_mean['streak'] / df_l3_away_mean['streak'])
			colnames.append("ratio_streak_l3")


			eachFeaturelist.append(df_mean_home['points'] / df_mean_away['points'])
			colnames.append("ratio_points_season")

			eachFeaturelist.append(df_l10_home_mean['points'] / df_l10_away_mean['points'])
			colnames.append("ratio_points_l10")

			eachFeaturelist.append(df_l5_home_mean['points'] / df_l5_away_mean['points'])
			colnames.append("ratio_points_l5")

			eachFeaturelist.append(df_l3_home_mean['points'] / df_l3_away_mean['points'])
			colnames.append("ratio_points_l3")


			eachFeaturelist.append((df_mean_home['points'] / df_mean_away['points'])/(df_l10_home_mean['points'] / df_l10_away_mean['points']))
			colnames.append("trend_points_season_l10_ratio")

			eachFeaturelist.append((df_l10_home_mean['points'] / df_l10_away_mean['points'])/(df_l3_home_mean['points'] / df_l3_away_mean['points']))
			colnames.append("trend_points_l10_l3_ratio")

			eachFeaturelist.append((df_mean_home['blocks'] / df_mean_away['blocks'])/(df_l10_home_mean['blocks'] / df_l10_away_mean['blocks']))
			colnames.append("ratio_blocks_season_l10")

			eachFeaturelist.append((df_mean_home['blocks'] / df_mean_away['blocks'])/(df_l5_home_mean['blocks'] / df_l5_away_mean['blocks']))
			colnames.append("ratio_blocks_season_l5")

			eachFeaturelist.append((df_mean_home['blocks'] / df_mean_away['blocks'])/(df_l3_home_mean['blocks'] / df_l3_away_mean['blocks']))
			colnames.append("ratio_blocks_season_l3")



			# point differential
			eachFeaturelist.append(rowvalue['points'] - rowvalue['o:points'])
			colnames.append("point_differential")

			# append inner list transform to list_transform
			features.append(eachFeaturelist)



# turn into pandas dataframe
df_transform = pd.DataFrame(features)

# add on column names
df_transform.columns = colnames

# save dataframe to csv
df_transform.to_csv('transformedData.csv', index=False)



