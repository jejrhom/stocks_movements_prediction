import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from finta import TA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import re
from bs4 import BeautifulSoup
import requests
from credentials import *
import tweepy
from textblob import TextBlob
from datetime import datetime, timedelta
import seaborn as sns
from nltk.corpus import stopwords

period_date = "5y"

def stock_movement(company):

    def companyname(company):
        todelete = ["Inc","Corporation", "LLC", "Corp"]
        name_company = yf.Ticker(company).info["shortName"]
        name_company = " ".join(re.findall("[a-zA-Z]+", name_company))
        for element in todelete :
            name_company = name_company.replace(element, "")
        return name_company

    def companysector(company):
        sector_company = "Multiple"
        if "sector" in yf.Ticker(company).info :
            sector_company = yf.Ticker(company).info["sector"]
        return sector_company

    def companylogo(company):
        logo_company = "https://cdn-prod.voxy.com/wp-content/uploads/2014/05/shrug-1.jpeg"
        if "logo_url" in yf.Ticker(company).info :
            logo_company = yf.Ticker(company).info["logo_url"]
        return logo_company

    def companyBusinessSummary(company):
        BusinessSummary_company = 0
        if "longBusinessSummary" in yf.Ticker(company).info :
            BusinessSummary_company = yf.Ticker(company).info["longBusinessSummary"]
        return BusinessSummary_company

    def companyearnings(company):
        earnings = yf.Ticker(company).earnings
        return earnings

    def analystsrecommendations(company):
        recommendations = yf.Ticker(company).recommendations
        return recommendations

    name_company = str(companyname(company))
    BusinessSummary_company = str(companyBusinessSummary(company))
    logo_company = str(companylogo(company))
    sector_company = str(companysector(company))

    def generate_urlNews(name_company):
        url = "https://news.search.yahoo.com/search?q={}".format(name_company)
        return url

    url_company = generate_urlNews(name_company)

    def scrap_news(url) :
        news = {"titles":[],"sources":[],"times":[], "times_hours":[],"links":[]}
        links_list = []
        time_hours = 0
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for news_item in soup.find_all('div', class_='NewsArticle'):
            title = news_item.find(['h4']).text
            time = news_item.find('span', class_='fc-2nd').text
            source = news_item.find("span", class_ ="s-source mr-5 cite-co").text
            link = news_item.find('a')["href"]
            # Clean time text and generate hours from publication to order the news in a timely manner
            time = time.replace('·', '').strip()
            if 'days' in time or 'day' in time :
                time_hours = int(time.split(" ")[0])*24
            else : 
                time_hours = int(time.split(" ")[0])
            news["titles"].append(title)
            news["sources"].append(source)
            news["times"].append(time)
            news["times_hours"].append(time_hours)
            news["links"].append(link)
        return news

    newsmetadata = pd.DataFrame(scrap_news(url_company)).sort_values("times_hours")
    newsmetadata['title_source']=newsmetadata['titles']+' - '+newsmetadata['sources']

    articles_dic = dict(zip(newsmetadata.title_source, newsmetadata.links))
    
    def stockPrice(company, period_date):
        data_stockcompany = yf.Ticker(str(company)).history(period=period_date)
        return data_stockcompany

    df = stockPrice(company, period_date)
    date = str(df[-1:].index[0]).split(" ")[0]
    open_price = df[-1:]["Open"][0]
    actual_price = df[-1:]["Close"][0]
    high_price = df[-1:]["High"][0]
    low_price = df[-1:]["Low"][0]
    volume_traded = df[-1:]["Volume"][0]

    diff_price = round(((df[-1:]["Close"][0]-df[-2:]["Close"][0])/df[-2:]["Close"][0])*100,2)


    def exponential_smooth(data, alpha):
        return data.ewm(alpha=alpha).mean()

    data = exponential_smooth(df, 0.65)

    data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
    indicators = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']

    def features_creation(data):
        for indicator in indicators:
            ind_data = eval('TA.' + indicator + '(data)')
            if not isinstance(ind_data, pd.DataFrame):
                ind_data = ind_data.to_frame()
            data = data.merge(ind_data, left_index=True, right_index=True)
        data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

        # Also calculate moving averages for features
        data['ema50'] = data['close'] / data['close'].ewm(50).mean()
        data['ema21'] = data['close'] / data['close'].ewm(21).mean()
        data['ema15'] = data['close'] / data['close'].ewm(14).mean()
        data['ema5'] = data['close'] / data['close'].ewm(5).mean()

        # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
        data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()
        
        return data.fillna(0)

    data_features = features_creation(data)

    def label_data(data, period = 15):
        prediction = (data.shift(-period)['close'] >= data['close'])
        prediction = prediction.iloc[:-period]
        data['pred'] = prediction.astype(int)
        return data

    labeled_data = label_data(data_features)

    def prediction_probability(labeled_data):
    
        #Data to input
        last_15days = labeled_data.tail(15)
        labeled_data.drop(df.tail(15).index, inplace=True)
    
        #Building model
        labels = labeled_data.pred
        features = labeled_data.drop(["pred","Dividends","Stock Splits","open", "high","low"], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state = 0)
        classifier = RandomForestClassifier(max_features = 'auto', n_estimators=1000, random_state=0)
        classifier.fit(x_train, y_train)
    
        #Compute predictions (x_test and last 15 days), probabilities and differences
        y_pred = classifier.predict(x_test)
        prediction = classifier.predict(last_15days.drop(["pred","Dividends","Stock Splits","open", "high","low"], axis=1))
        probabilities = classifier.predict_proba(last_15days.drop(["pred","Dividends","Stock Splits","open", "high","low"], axis=1))
        delta_buy = (probabilities[-1][1] - probabilities[-2][1])/probabilities[-2][1]
        if delta_buy > 0 :
            delta_buy = "+"+ str(round(delta_buy*100,2))
        else :
            delta_buy = str(round(delta_buy*100,2))
        delta_sell = (probabilities[-1][0] - probabilities[-2][0])/probabilities[-2][0]
        if delta_sell > 0 :
            delta_sell = "+"+ str(round(delta_sell*100,2))
        else :
            delta_sell = str(round(delta_sell*100,2))
        probabilities_list = [probabilities[-1][1], delta_buy, probabilities[-1][0], delta_sell]
        return probabilities_list

    probabilities = prediction_probability(labeled_data)

    
    results = [logo_company, name_company,  sector_company, BusinessSummary_company, date, round(open_price,2), round(high_price,2), round(low_price,2), volume_traded, probabilities[0],probabilities[1], probabilities[2], probabilities[3], articles_dic, str(diff_price), round(actual_price,2)]
    return results

