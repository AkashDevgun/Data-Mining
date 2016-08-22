from csv import DictReader, DictWriter
from bs4 import BeautifulSoup
import urllib2
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


links = [
"http://sportsdatabase.com/nba/query?output=default&sdql=date%2C+team%2C+assists%2C+ats+margin%2C+ats+streak%2C+biggest+lead%2C+blocks%2C+conference%2C+defensive+rebounds%2C+fast+break+points%2C+field+goals+attempted%2C+field+goals+made%2C+fouls%2C+free+throws+attempted%2C+free+throws+made%2C+game+number%2C+line%2C+losses%2C+margin%2C+margin+after+the+first%2C+margin+after+the+third%2C+margin+at+the+half%2C+matchup+losses%2C+matchup+wins%2C+offensive+rebounds%2C+playoffs%2C+points%2C+points+in+the+paint%2C+position%2C+rebounds%2C+rest%2C+season%2C+steals%2C+streak%2C+team+rebounds%2C+three+pointers+attempted%2C+three+pointers+made%2C+total%2C+turnovers%2C+wins%2C+o%3Ateam%2C+o%3Aassists%2C+o%3Aats+margin%2C+o%3Aats+streak%2C+o%3Abiggest+lead%2C+o%3Ablocks%2C+o%3Aconference%2C+o%3Adefensive+rebounds%2C+o%3Afast+break+points%2C+o%3Afield+goals+attempted%2C+o%3Afield+goals+made%2C+o%3Afouls%2C+o%3Afree+throws+attempted%2C+o%3Afree+throws+made%2C+o%3Agame+number%2C+o%3Aline%2C+o%3Alosses%2C+o%3Amargin%2C+o%3Amargin+after+the+first%2C+o%3Amargin+after+the+third%2C+o%3Amargin+at+the+half%2C+o%3Amatchup+losses%2C+o%3Amatchup+wins%2C+o%3Aoffensive+rebounds%2C+o%3Aplayoffs%2C+o%3Apoints%2C+o%3Apoints+in+the+paint%2C+o%3Aposition%2C+o%3Arebounds%2C+o%3Arest%2C+o%3Aseason%2C+o%3Asteals%2C+o%3Astreak%2C+o%3Ateam+rebounds%2C+o%3Athree+pointers+attempted%2C+o%3Athree+pointers+made%2C+o%3Atotal%2C+o%3Aturnovers%2C+o%3Awins+%40+season%3D2010&submit=++S+D+Q+L+!++",
"http://sportsdatabase.com/nba/query?output=default&sdql=date%2C+team%2C+assists%2C+ats+margin%2C+ats+streak%2C+biggest+lead%2C+blocks%2C+conference%2C+defensive+rebounds%2C+fast+break+points%2C+field+goals+attempted%2C+field+goals+made%2C+fouls%2C+free+throws+attempted%2C+free+throws+made%2C+game+number%2C+line%2C+losses%2C+margin%2C+margin+after+the+first%2C+margin+after+the+third%2C+margin+at+the+half%2C+matchup+losses%2C+matchup+wins%2C+offensive+rebounds%2C+playoffs%2C+points%2C+points+in+the+paint%2C+position%2C+rebounds%2C+rest%2C+season%2C+steals%2C+streak%2C+team+rebounds%2C+three+pointers+attempted%2C+three+pointers+made%2C+total%2C+turnovers%2C+wins%2C+o%3Ateam%2C+o%3Aassists%2C+o%3Aats+margin%2C+o%3Aats+streak%2C+o%3Abiggest+lead%2C+o%3Ablocks%2C+o%3Aconference%2C+o%3Adefensive+rebounds%2C+o%3Afast+break+points%2C+o%3Afield+goals+attempted%2C+o%3Afield+goals+made%2C+o%3Afouls%2C+o%3Afree+throws+attempted%2C+o%3Afree+throws+made%2C+o%3Agame+number%2C+o%3Aline%2C+o%3Alosses%2C+o%3Amargin%2C+o%3Amargin+after+the+first%2C+o%3Amargin+after+the+third%2C+o%3Amargin+at+the+half%2C+o%3Amatchup+losses%2C+o%3Amatchup+wins%2C+o%3Aoffensive+rebounds%2C+o%3Aplayoffs%2C+o%3Apoints%2C+o%3Apoints+in+the+paint%2C+o%3Aposition%2C+o%3Arebounds%2C+o%3Arest%2C+o%3Aseason%2C+o%3Asteals%2C+o%3Astreak%2C+o%3Ateam+rebounds%2C+o%3Athree+pointers+attempted%2C+o%3Athree+pointers+made%2C+o%3Atotal%2C+o%3Aturnovers%2C+o%3Awins+%40+season%3D2011&submit=++S+D+Q+L+!++",
"http://sportsdatabase.com/nba/query?output=default&sdql=date%2C+team%2C+assists%2C+ats+margin%2C+ats+streak%2C+biggest+lead%2C+blocks%2C+conference%2C+defensive+rebounds%2C+fast+break+points%2C+field+goals+attempted%2C+field+goals+made%2C+fouls%2C+free+throws+attempted%2C+free+throws+made%2C+game+number%2C+line%2C+losses%2C+margin%2C+margin+after+the+first%2C+margin+after+the+third%2C+margin+at+the+half%2C+matchup+losses%2C+matchup+wins%2C+offensive+rebounds%2C+playoffs%2C+points%2C+points+in+the+paint%2C+position%2C+rebounds%2C+rest%2C+season%2C+steals%2C+streak%2C+team+rebounds%2C+three+pointers+attempted%2C+three+pointers+made%2C+total%2C+turnovers%2C+wins%2C+o%3Ateam%2C+o%3Aassists%2C+o%3Aats+margin%2C+o%3Aats+streak%2C+o%3Abiggest+lead%2C+o%3Ablocks%2C+o%3Aconference%2C+o%3Adefensive+rebounds%2C+o%3Afast+break+points%2C+o%3Afield+goals+attempted%2C+o%3Afield+goals+made%2C+o%3Afouls%2C+o%3Afree+throws+attempted%2C+o%3Afree+throws+made%2C+o%3Agame+number%2C+o%3Aline%2C+o%3Alosses%2C+o%3Amargin%2C+o%3Amargin+after+the+first%2C+o%3Amargin+after+the+third%2C+o%3Amargin+at+the+half%2C+o%3Amatchup+losses%2C+o%3Amatchup+wins%2C+o%3Aoffensive+rebounds%2C+o%3Aplayoffs%2C+o%3Apoints%2C+o%3Apoints+in+the+paint%2C+o%3Aposition%2C+o%3Arebounds%2C+o%3Arest%2C+o%3Aseason%2C+o%3Asteals%2C+o%3Astreak%2C+o%3Ateam+rebounds%2C+o%3Athree+pointers+attempted%2C+o%3Athree+pointers+made%2C+o%3Atotal%2C+o%3Aturnovers%2C+o%3Awins+%40+season%3D2012&submit=++S+D+Q+L+!++",
"http://sportsdatabase.com/nba/query?output=default&sdql=date%2C+team%2C+assists%2C+ats+margin%2C+ats+streak%2C+biggest+lead%2C+blocks%2C+conference%2C+defensive+rebounds%2C+fast+break+points%2C+field+goals+attempted%2C+field+goals+made%2C+fouls%2C+free+throws+attempted%2C+free+throws+made%2C+game+number%2C+line%2C+losses%2C+margin%2C+margin+after+the+first%2C+margin+after+the+third%2C+margin+at+the+half%2C+matchup+losses%2C+matchup+wins%2C+offensive+rebounds%2C+playoffs%2C+points%2C+points+in+the+paint%2C+position%2C+rebounds%2C+rest%2C+season%2C+steals%2C+streak%2C+team+rebounds%2C+three+pointers+attempted%2C+three+pointers+made%2C+total%2C+turnovers%2C+wins%2C+o%3Ateam%2C+o%3Aassists%2C+o%3Aats+margin%2C+o%3Aats+streak%2C+o%3Abiggest+lead%2C+o%3Ablocks%2C+o%3Aconference%2C+o%3Adefensive+rebounds%2C+o%3Afast+break+points%2C+o%3Afield+goals+attempted%2C+o%3Afield+goals+made%2C+o%3Afouls%2C+o%3Afree+throws+attempted%2C+o%3Afree+throws+made%2C+o%3Agame+number%2C+o%3Aline%2C+o%3Alosses%2C+o%3Amargin%2C+o%3Amargin+after+the+first%2C+o%3Amargin+after+the+third%2C+o%3Amargin+at+the+half%2C+o%3Amatchup+losses%2C+o%3Amatchup+wins%2C+o%3Aoffensive+rebounds%2C+o%3Aplayoffs%2C+o%3Apoints%2C+o%3Apoints+in+the+paint%2C+o%3Aposition%2C+o%3Arebounds%2C+o%3Arest%2C+o%3Aseason%2C+o%3Asteals%2C+o%3Astreak%2C+o%3Ateam+rebounds%2C+o%3Athree+pointers+attempted%2C+o%3Athree+pointers+made%2C+o%3Atotal%2C+o%3Aturnovers%2C+o%3Awins+%40+season%3D2013&submit=++S+D+Q+L+!++",
"http://sportsdatabase.com/nba/query?output=default&sdql=date%2C+team%2C+assists%2C+ats+margin%2C+ats+streak%2C+biggest+lead%2C+blocks%2C+conference%2C+defensive+rebounds%2C+fast+break+points%2C+field+goals+attempted%2C+field+goals+made%2C+fouls%2C+free+throws+attempted%2C+free+throws+made%2C+game+number%2C+line%2C+losses%2C+margin%2C+margin+after+the+first%2C+margin+after+the+third%2C+margin+at+the+half%2C+matchup+losses%2C+matchup+wins%2C+offensive+rebounds%2C+playoffs%2C+points%2C+points+in+the+paint%2C+position%2C+rebounds%2C+rest%2C+season%2C+steals%2C+streak%2C+team+rebounds%2C+three+pointers+attempted%2C+three+pointers+made%2C+total%2C+turnovers%2C+wins%2C+o%3Ateam%2C+o%3Aassists%2C+o%3Aats+margin%2C+o%3Aats+streak%2C+o%3Abiggest+lead%2C+o%3Ablocks%2C+o%3Aconference%2C+o%3Adefensive+rebounds%2C+o%3Afast+break+points%2C+o%3Afield+goals+attempted%2C+o%3Afield+goals+made%2C+o%3Afouls%2C+o%3Afree+throws+attempted%2C+o%3Afree+throws+made%2C+o%3Agame+number%2C+o%3Aline%2C+o%3Alosses%2C+o%3Amargin%2C+o%3Amargin+after+the+first%2C+o%3Amargin+after+the+third%2C+o%3Amargin+at+the+half%2C+o%3Amatchup+losses%2C+o%3Amatchup+wins%2C+o%3Aoffensive+rebounds%2C+o%3Aplayoffs%2C+o%3Apoints%2C+o%3Apoints+in+the+paint%2C+o%3Aposition%2C+o%3Arebounds%2C+o%3Arest%2C+o%3Aseason%2C+o%3Asteals%2C+o%3Astreak%2C+o%3Ateam+rebounds%2C+o%3Athree+pointers+attempted%2C+o%3Athree+pointers+made%2C+o%3Atotal%2C+o%3Aturnovers%2C+o%3Awins+%40+season%3D2014&submit=++S+D+Q+L+!++",
"http://sportsdatabase.com/nba/query?output=default&sdql=date%2C+team%2C+assists%2C+ats+margin%2C+ats+streak%2C+biggest+lead%2C+blocks%2C+conference%2C+defensive+rebounds%2C+fast+break+points%2C+field+goals+attempted%2C+field+goals+made%2C+fouls%2C+free+throws+attempted%2C+free+throws+made%2C+game+number%2C+line%2C+losses%2C+margin%2C+margin+after+the+first%2C+margin+after+the+third%2C+margin+at+the+half%2C+matchup+losses%2C+matchup+wins%2C+offensive+rebounds%2C+playoffs%2C+points%2C+points+in+the+paint%2C+position%2C+rebounds%2C+rest%2C+season%2C+steals%2C+streak%2C+team+rebounds%2C+three+pointers+attempted%2C+three+pointers+made%2C+total%2C+turnovers%2C+wins%2C+o%3Ateam%2C+o%3Aassists%2C+o%3Aats+margin%2C+o%3Aats+streak%2C+o%3Abiggest+lead%2C+o%3Ablocks%2C+o%3Aconference%2C+o%3Adefensive+rebounds%2C+o%3Afast+break+points%2C+o%3Afield+goals+attempted%2C+o%3Afield+goals+made%2C+o%3Afouls%2C+o%3Afree+throws+attempted%2C+o%3Afree+throws+made%2C+o%3Agame+number%2C+o%3Aline%2C+o%3Alosses%2C+o%3Amargin%2C+o%3Amargin+after+the+first%2C+o%3Amargin+after+the+third%2C+o%3Amargin+at+the+half%2C+o%3Amatchup+losses%2C+o%3Amatchup+wins%2C+o%3Aoffensive+rebounds%2C+o%3Aplayoffs%2C+o%3Apoints%2C+o%3Apoints+in+the+paint%2C+o%3Aposition%2C+o%3Arebounds%2C+o%3Arest%2C+o%3Aseason%2C+o%3Asteals%2C+o%3Astreak%2C+o%3Ateam+rebounds%2C+o%3Athree+pointers+attempted%2C+o%3Athree+pointers+made%2C+o%3Atotal%2C+o%3Aturnovers%2C+o%3Awins+%40+season%3D2015&submit=++S+D+Q+L+!++"]

count = 1

counter = 0

for link in links:
	hdr={'User-Agent':'Mozilla/5.0'} 
	req = urllib2.Request(link,headers = hdr)


	pagedata = urllib2.urlopen(req)


	#pagedata = open("output.txt",'r').read()

	soup = BeautifulSoup(pagedata)

	columndata = soup.find("table", {"border":"0", "id":"DT_Table"})

	subdata = columndata.find("thead").find("tr").findAll("th")


	columnare = []

	if count == 1:
		for metadata in subdata:
			columnare.append(metadata.string.strip())
		print columnare
		count = 2
		dataF = pd.DataFrame(columns = columnare)
	

	tabledata = soup.find("table", {"border":"0", "id":"DT_Table"})

	againdata = tabledata.findAll("tr")

	#,{"bgcolor":"ffffff", "valign":"top"}
	



	for row in againdata:
		allcolumns = row.findAll("td")
		arr = []
		if len(allcolumns) > 0:
			for colum in allcolumns:
				val = colum.string.strip()
				arr.append(val)
			dataF.loc[counter] = arr 
			counter = counter + 1


	print len(np.unique(dataF['team']))

dataF.to_csv('AllSeasonsData.csv', sep = '\t',index=False)

