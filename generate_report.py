#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import timedelta, datetime
from pandas.plotting import table
from matplotlib.dates import DateFormatter
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import pylab
import dataframe_image as dfi

# %matplotlib inline


# In[2]:


class StringConverter(dict):
    def __contains__(self, item):
        return True

    def __getitem__(self, item):
        return str

    def get(self, default=None):
        return str


# In[3]:


from string import ascii_lowercase

alphabetlist = []

for alpha in ascii_lowercase:
    for bravo in ascii_lowercase:
        for charlie in ascii_lowercase:
            alphabetlist.append("{0}{1}{2}".format(alpha,bravo,charlie))


# In[4]:


reportMonYear = "October - 2020"


# In[5]:


sml = pd.read_csv("seas_msg_log.csv", converters=StringConverter())


# In[6]:


# create new column to merge with comsat
sml['basecode'] = sml['filename'].str.slice(0,-4,1)


# In[7]:


# because of potential user error, need to use processed datetime
sml['processed'] = pd.to_datetime(sml['processed'],utc=True)
sml['rec_date'] = pd.to_datetime(sml['processed'],utc=True).dt.strftime('%Y-%m-%d')
sml['rec_time'] = pd.to_datetime(sml['processed'],utc=True).dt.strftime('%H%M')
sml['msg_date'] = pd.to_datetime(sml['msg_date'],utc=True)


# In[8]:


sml['msgtype'] = sml['filename'].str.slice(0,2,1)


# In[9]:


sml['recorded_date'] = sml['rec_date'] + " " + sml['rec_time'].str.slice(0,2,1) + ":" + sml['rec_time'].str.slice(2,4,1)
sml['recorded_date'] = pd.to_datetime(sml['recorded_date'],utc=True)


# In[10]:


com = pd.read_csv("comsat.csv", converters=StringConverter())


# In[11]:


com['comsat_datetime'] = pd.to_datetime(com['comsat_datetime'])


# In[12]:


com['lloyds'] = pd.to_numeric(com['lloyds'])


# In[13]:


com['rec_time'] = com['rec_time'].apply(str)


# In[14]:


# left join tables
mergeSmlCom = pd.merge(sml, com, on=['basecode','rec_date','rec_time'], how='left')


# In[15]:


# get rows that have comsat info. Rows were processed
processed = mergeSmlCom[~mergeSmlCom['wmo_id'].isnull()]


# In[16]:


# get unprocessed messages data
unprocessed = mergeSmlCom[mergeSmlCom['wmo_id'].isnull()]


# In[17]:


# remove older comsat data that don't match seas msg timestamp
rmvOldComsat = processed[processed['comsat_datetime']-processed['processed'] > timedelta(days=0)]


# In[18]:


# remove messages join with two different rows of comsat data; select earliest date
# caution that if two filenames are the same and comsat data is badly delay, the two filenames will have the same comsat data
# in order for that to happened, some parts of the seas process will not be running for about a month
rmvOutOfRngDate = pd.merge(rmvOldComsat, rmvOldComsat.groupby(['filename','processed'], as_index=False)['comsat_datetime'].max(), on=['filename','processed','comsat_datetime'])


# In[19]:


# remove duplicates
rmvDups = rmvOutOfRngDate.drop_duplicates(['filename','processed'],keep='first')


# In[20]:


gts = pd.read_csv("gts_nws_distribute.csv")


# In[21]:


gts['basecode'] = gts['gts_filename'].str.slice(0,-4,1)


# In[22]:


#deliveryDT = pd.to_datetime(missingMsg['delivery_datetime'])
#deliveryDT.dt.date.value_counts()
#missingMsg = completeMsgTable[completeMsgTable['filename'].isnull()]
#missingMsg['gtstype'].value_counts()
gts['delivery_datetime'] = pd.to_datetime(gts['delivery_datetime'],utc=True)


# In[23]:


# full outer join
allTables = pd.merge(rmvDups, gts, on='basecode',how='left')


# In[24]:


allTables['processing_time'] = allTables['comsat_datetime'] - allTables['processed']


# In[25]:


allTables['to_gts_time'] = allTables['delivery_datetime'] - allTables['processed']


# In[26]:


allTables['from_obs_to_gts_time'] = allTables['delivery_datetime'] - allTables['recorded_date']


# In[27]:


allTables['msg_date_to_gts_time'] = allTables['delivery_datetime'] - allTables['msg_date']


# In[28]:


#completeTable[(completeTable['msg_date'] > '2020-02-28 00:00:00') & (completeTable['msg_date'] <= '2020-02-29 23:59:59')]
allTables['to_gts_time_num'] = allTables['to_gts_time'].values.astype(np.int64)
allTables['processing_time_num'] = allTables['processing_time'].values.astype(np.int64)
allTables['from_obs_to_gts_time_num'] = allTables['from_obs_to_gts_time'].values.astype(np.int64)


# In[29]:


deliveredMsg = allTables[~allTables['delivery_datetime'].isnull()]


# In[30]:


# If delivery_date was weeks after processed date then it is the following file with same filename
# If delivery_date was before processed date then it was an older file
completeTable = deliveredMsg[(deliveredMsg['delivery_datetime']-deliveredMsg['processed'] < timedelta(days=2)) & (deliveredMsg['delivery_datetime']-deliveredMsg['processed'] > timedelta(days=0))]


# In[31]:


ctMeanResults = completeTable.groupby(completeTable['processed'].dt.date).mean()
toGtsTimeAvgDays = pd.to_timedelta(ctMeanResults['to_gts_time_num']).astype('timedelta64[m]').rename('Avg. Time To GTS')
processTimeAvgDays = pd.to_timedelta(ctMeanResults['processing_time_num']).astype('timedelta64[m]').rename('Avg. Time To Process')
obsToGtsTimeAvgDays = pd.to_timedelta(ctMeanResults['from_obs_to_gts_time_num']).astype('timedelta64[m]').rename('Avg. Time From Obs. To GTS (Mins)')
avgTimesDays = pd.concat([processTimeAvgDays, toGtsTimeAvgDays], axis=1)


# In[32]:


avgTimesDays.index.name = None


# In[33]:


processedCountByDay = completeTable.groupby(completeTable['processed'].dt.date).count()['filename'].rename('Messages Received')


# In[34]:


deliveryCountByDay = completeTable.groupby(completeTable['delivery_datetime'].dt.date).count()['filename'].rename('Messages Sent To GTS')


# In[35]:


procDelCount = pd.concat([processedCountByDay, deliveryCountByDay], axis=1)


# In[36]:


procDelCount = procDelCount.fillna(0)


# In[37]:


procDelCount.iloc[-1]['Messages Received'] = np.nan


# In[38]:


#Get only messages that were meant to be processed but didn't instead of fake messages
unprocessableMsg = unprocessed[unprocessed['filename'].str.contains('.txt')]


# In[39]:


#remove code43 messages from calculations
rmvMsgCode43 = unprocessableMsg[unprocessableMsg['msgtype'] != '01']
dateunprocess1 = rmvMsgCode43.groupby(rmvMsgCode43['processed'].dt.date).count()['filename'].rename('NoContent')


# In[40]:


#allTables[allTables['delivery_datetime'].isnull()]['msgtype'] != '01'
didnotGTS = allTables[allTables['delivery_datetime'].isnull()]
#remove code43 messages from calculations
rmvCode43 = didnotGTS[didnotGTS['msgtype'] != '01']
dateunprocess2 = rmvCode43.groupby(rmvCode43['processed'].dt.date).count()['filename'].rename('ContentUnableParsing')


# In[41]:


conUnprocess = pd.concat([dateunprocess1, dateunprocess2], axis=1).sort_index()


# In[42]:


prepUnprocess = conUnprocess.fillna(0)
prepUnprocess['Unprocessable Messages'] = prepUnprocess['ContentUnableParsing'] + prepUnprocess['NoContent']
UnprocessCountByDate = prepUnprocess.drop(['ContentUnableParsing', 'NoContent'], axis=1)


# In[43]:


procDelCount['Messages Received'] = procDelCount['Messages Received'] + UnprocessCountByDate['Unprocessable Messages']


# In[44]:


unprocessableCount = UnprocessCountByDate['Unprocessable Messages'].sum()


# In[45]:


processableCount = completeTable['wmo_id'].value_counts().sum()


# In[46]:


fig = plt.figure(figsize=(21, 18))
plt.suptitle("SEAS VOS Report - {0}".format(reportMonYear), fontsize=25)
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

fax1 = fig.add_subplot(spec[0, :])
#fig, ax = plt.subplots(1, 1,figsize=(12, 7))
#table(fax1, np.round(procDelCount.describe(), 2),loc='upper center', colWidths=[0.2, 0.2])
fax1.set_title("Number of Messages Received And Sent To GTS\n{0} - {1}"
               .format(procDelCount.index.min().strftime("%d %B, %Y"),
                       procDelCount.index.max().strftime("%d %B, %Y")),
               fontsize=15)
fax1.xaxis.set_major_formatter(DateFormatter("%m-%d"))
fax1.set_ylabel('Number of Messages', fontsize=14)
fax1.tick_params(direction='inout', length=6, width=2,labelsize='large')

procDelCount.plot(ax=fax1, rot=45)

fax2 = fig.add_subplot(spec[1, 0])
#fig1, ax1 = plt.subplots(1, 1,figsize=(7, 4))
fax2.xaxis.set_major_formatter(DateFormatter("%m-%d"))
fax2.tick_params(direction='inout', length=4, width=3,labelsize='large')
fax2.set_title("Average Times For Messages To Process And For Messages To Be Sent To GTS\n{0} - {1}"
               .format(avgTimesDays.index.min().strftime("%d %B, %Y"),
                       avgTimesDays.index.max().strftime("%d %B, %Y")),
               fontsize=15)
#ax1.set_xlabel('Month - Day')
fax2.set_ylabel('Minutes', fontsize=14)
avgTimesDays.plot(ax=fax2, rot=60)

fax3 = fig.add_subplot(spec[1, 1])
countSizes = [unprocessableCount, processableCount]
labelProcess = ['Messages that were not able to be Processed', 'Messages that were able to be Processed']
#fig1, ax1 = plt.subplots(figsize=(6,6))
fax3.pie(countSizes, labels=labelProcess, autopct='%1.1f%%',
         startangle=270, textprops={'fontsize':15})
fax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('messages_data_info.png')


# In[47]:


sumCounts = pd.concat([procDelCount, UnprocessCountByDate], axis=1)


# In[48]:


sumCounts['Unprocessable Percent'] = ((sumCounts['Unprocessable Messages'] / sumCounts['Messages Received']) * 100).round(2)


# In[49]:


summaryDetailByDay = pd.concat([sumCounts, obsToGtsTimeAvgDays], axis=1).reset_index().rename(columns={"processed":"Date","Unprocessable Messages":"Unprocessable Msg.","Messages Received":"Msg. Received","Messages Sent To GTS":"Msg. Sent To GTS"}).applymap(str)


# In[50]:


dfi.export(summaryDetailByDay,"day_by_day_summary.png")


# In[51]:


#wmoidMeanResults = completeTable.groupby(completeTable["wmo_id"]).mean()
#wmoidAvgObsToGts = pd.to_timedelta(wmoidMeanResults['from_obs_to_gts_time_num']).astype('timedelta64[m]').rename('Avg. Time From Obs. To GTS (Mins)').reset_index()
wmoidCounts = completeTable['wmo_id'].value_counts().rename('Total Received').reset_index().rename(columns={"index":"wmo_id"})


# In[52]:


avgMsgSentPerDay = completeTable.groupby([completeTable['processed'].dt.date,'wmo_id']).count()['filename'].reset_index().groupby(['wmo_id']).mean().rename(columns={"filename":"Avg. Msg. Sent/Day"}).round(2).reset_index()


# In[53]:


wmoIDsummaryDetail = pd.merge(wmoidCounts, avgMsgSentPerDay, on=['wmo_id'], how='inner').sort_values(by=['Total Received'], ascending=False).applymap(str)


# In[54]:


alphabetlist
alphabetcount = 0
lenwmoIDSD = len(wmoIDsummaryDetail)
stepNum = 30
for lastp in np.arange(stepNum,lenwmoIDSD,stepNum):
    firstp = lastp-stepNum
    dftemp = wmoIDsummaryDetail[firstp:lastp]
    pngname = "wmo_id_summary_{0}.png".format(alphabetlist[alphabetcount])
    alphabetcount += 1
    dfi.export(dftemp,pngname)
    
    if ((lastp+stepNum) > lenwmoIDSD):
        dftemp = wmoIDsummaryDetail[lastp:lenwmoIDSD]
        pngname = "wmo_id_summary_{0}.png".format(alphabetlist[alphabetcount])
        dfi.export(dftemp,pngname)


# In[ ]:


callsignsLoc = {}

for index, row in completeTable.iterrows():
    wmoID = row['wmo_id']
    if wmoID in callsignsLoc:
        callsignsLoc[wmoID].append([row['latitude'],row['longitude']])
    else:
        callsignsLoc[wmoID] = [[row['latitude'],row['longitude']]]

msgCount = processedCountByDay.sum()
shpCount = len(completeTable['wmo_id'].value_counts())

fig = plt.figure(figsize=(12, 15))
plt.suptitle("SEAS VOS Report - {0}".format(reportMonYear), fontsize=25)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.stock_img()

ax.set_title("Number of Ships: {0} - Number of Messages: {1}".format(shpCount, msgCount), fontsize=30)
#completeTable[['wmo_id','latitude','longitude']]
#mpatchesList = []
#callsignList = []

for wmoidKey, latlonList in callsignsLoc.items():
    curcolor = np.random.rand(3,) #next(multicolors)
    #callsignList.append(key)
    #mpatchesList.append(mpatches.Circle((0, 0), facecolor=curcolor))
    for latlon in latlonList:
        longitude = [float(latlon[1])]
        latitude = [float(latlon[0])]
        plt.plot(longitude, latitude,color=curcolor, linewidth=1, marker='.',transform=ccrs.Geodetic(), alpha=0.75)
#plt.legend(mpatchesList, callsignList, loc='upper center', bbox_to_anchor=(0.5, -0.05),fontsize='medium', fancybox=True, shadow=True, ncol=10)
plt.savefig('callsigns_map.png')


# In[ ]:


#completeTable['to_gts_time'].mean()
#completeTable['processing_time'].mean()
#wmo_id_counts = completeTable['wmo_id'].value_counts()
#wmo_id_counts.std() * 3
#wmo_id_percents = np.round(completeTable['wmo_id'].value_counts(normalize=True) * 100,2)
#callsignCountPercent = pd.concat([wmo_id_counts, wmo_id_percents], axis=1)
#callsignCountPercent.to_csv("callsign_count_percent.csv")
#numOfShips = len(wmo_id_counts)
#msgReceived = processedCountByDay.sum()
#wmoIdTopTen = wmo_id_counts.head()
#wmoIdTopHalf = wmo_id_counts.head(int(len(wmo_id_counts)/2))
#shipCallSignIndexed = completeTable.set_index(['wmo_id','shipname'])
#scsindexCount = shipCallSignIndexed.index.value_counts().rename('Count')
#scsindexPercent = shipCallSignIndexed.index.value_counts(normalize=True).rename('Percentage')
#scsindexOutput = pd.concat([scsindexCount, scsindexPercent], axis=1)
#msgReceived, unprocessableCount

