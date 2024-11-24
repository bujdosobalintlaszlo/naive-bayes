#-----------------------------------------------
#Importok
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
#-----------------------------------------------

#-----------------------------------------------
# Spam-es szavak listaja
advertising_keywords = [
    'sale', 'discount', 'buy', 'offer', 'deal', 'free', 'limited', 'now', 
    'save', 'exclusive', 'bargain', 'special', 'new', 'best', 'only', 'hurry', 
    'today', 'shop', 'shopnow', 'clearance', 'lowest', 'reduced', 'flashsale', 
    'gift', 'bonus', 'unmissable', 'hot', 'musthave', 'unbeatable', 'grab', 
    'promotion', 'lastchance', 'topdeal', 'extra', 'bonus', 'cashback', 'guarantee', 
    'instant', 'discounted', 'freegift', 'blackfriday', 'cybermonday', 'limitedtime', 
    'earlybird', 'dealofday', 'stockclearance', 'off', 'pricecut', 'specialoffer', 
    'bigdiscount', 'exclusiveoffer', 'shoppingdeal', 'todayonly'
]
#-----------------------------------------------

#-----------------------------------------------
# szoismetles
def word_repetition(text):
    words = text.lower().split()
    word_counts = pd.Series(words).value_counts()
    return word_counts.to_dict()
#-----------------------------------------------

#-----------------------------------------------
#megnezei h vannak e benne olyan szavak amik spam gyanusak 'advertising_keywords'
def keyword_frequency(text):
    keywords_count = 0
    words = text.lower().split()
    for word in advertising_keywords:
        keywords_count += words.count(word)
    return keywords_count
#-----------------------------------------------

#-----------------------------------------------
def text_length(text):
    return len(text)

#-----------------------------------------------

#-----------------------------------------------
# Megekeresi a 'to' szot es megnezzi h hany cimzett van
def multiple_recipients(text):
    match = re.search(r"To:\s*(.*)", text)
    
    if match:
        recipients = match.group(1)
        recipients_list = re.split(r",|and", recipients)
        recipients_list = [recipient.strip() for recipient in recipients_list if recipient.strip()]
        return len(recipients_list) > 1
    
    return False
#-----------------------------------------------

#-----------------------------------------------
#Megnezi van-e html content benne
def has_html_content(text):
    return bool(re.search(r'<[a-z][\s\S]*>', text))
#-----------------------------------------------

#-----------------------------------------------
#irasjelek figyelese
def sentence_analysis(text):
    sentences = re.split(r'[.!?]', text)
    return len(sentences)
#-----------------------------------------------

#-----------------------------------------------
# linkek szamanak figyelese
def link_count(text):
    return len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
#-----------------------------------------------
#specialis karakterek figyelese
def special_character_count(text):
    return len(re.findall(r'[^\w\s]', text))
#-----------------------------------------------

#-----------------------------------------------
#html formazasi tage checkolasa
def is_formatted(text):
    return bool(re.search(r'<(b|i)>.*</\1>', text))
#-----------------------------------------------

#-----------------------------------------------
# csatolmanyok megnezese (van/nincs)
def has_attachment(text):
    return bool(re.search(r'Attachment:.*', text))
#-----------------------------------------------

#-----------------------------------------------
# Olvashatosag (atlag szohossz)
def text_readability(text):
    words = text.split()
    avg_word_length = np.mean([len(word) for word in words])
    return avg_word_length
#-----------------------------------------------

#-----------------------------------------------
# pozitiv negativ szavak aranya
def sentiment_score(text):
    positive_words = ['good', 'great', 'amazing', 'excellent', 'fantastic']
    negative_words = ['bad', 'horrible', 'terrible', 'poor', 'awful']
    positive_count = sum([text.lower().count(word) for word in positive_words])
    negative_count = sum([text.lower().count(word) for word in negative_words])
    score = positive_count - negative_count
    return max(score, 0) 
#-----------------------------------------------

#-----------------------------------------------
# szovegformazaottsag
def readability_score(text):
    sentences = re.split(r'[.!?]', text)
    words = text.split()
    syllables = sum([len(re.findall(r'[aeiouAEIOU]', word)) for word in words])
    return 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
#-----------------------------------------------

#-----------------------------------------------
# emailek es azok spam/nem spma erteke (0,1)
emails = [
    """From: phillip.allen@enron.com
To: tim.belden@enron.com
Subject: Hurry! Get your discount today, limited time offer.

Dear Tim and Joe
Hurry and get your discount today! Buy now to save big.
http://www.sale.com/discount
Best, Phillip""",
    
    """From: sarah.jones@example.com
To: john.smith@example.com
Subject: Free gift with every purchase. Don't miss out!

Hey John,
We've got a special offer just for you! Get a free gift with every purchase.
Visit us at: http://www.giftshop.com/freegift
Cheers, Sarah""",
    
    """From: support@newsletter.com
To: user@example.com
Subject: Meeting tomorrow at 10 AM in the office.

Hello,
Just a <h1>reminder</h1> about tomorrow's meeting. See you at 10 AM in the office.
Best regards,
Support""",
    
    """From: deals@shopping.com
To: jane.doe@example.com
Subject: Sale ends soon! Get your discount on all items today.

Dear Jane,
The sale ends soon! Don't miss out on your chance to get discounts on everything.
Shop now: http://www.shopping.com/deals
Cheers, Deals Team""",
    
    """From: alerts@technews.com
To: tech.enthusiast@example.com, tech.enthusiast@example.com
Subject: Check out the new arrivals on our site. Big discounts available.

Hi,
New products are available now! Big discounts on all items in our new arrivals section.
Shop here: http://www.technews.com/new-arrivals
Best, Tech News Team""",
    
    """From: admin@website.com
To: user@example.com
Subject: Your invoice is ready for download. Please find the attachment.

Dear User,
Your invoice is ready for download. Please find the attachment for details.
Attachment: invoice_12345.pdf
Best regards,
Admin"""
]

labels = [1, 1, 0, 1, 1, 0]
#-----------------------------------------------

#-----------------------------------------------
# labelek es emailek szama egyezik-e ha nem kiirja es konnyen modosithato
if len(labels) != len(emails):
    print(f"Mismatch: {len(labels)} labels and {len(emails)} messages. Adjusting labels...")
    labels = labels[:len(emails)]
#-----------------------------------------------

#-----------------------------------------------
# dataframe-e konvertalas
data = pd.DataFrame({
    'email': emails,
    'label': labels
})
#-----------------------------------------------

#-----------------------------------------------
# adatok kinyerese
data['word_repetition'] = data['email'].apply(word_repetition)
data['keyword_frequency'] = data['email'].apply(keyword_frequency)
data['text_length'] = data['email'].apply(text_length)
data['multiple_recipients'] = data['email'].apply(multiple_recipients)
data['has_html_content'] = data['email'].apply(has_html_content)
data['sentence_count'] = data['email'].apply(sentence_analysis)
data['link_count'] = data['email'].apply(link_count)
data['special_character_count'] = data['email'].apply(special_character_count)
data['is_formatted'] = data['email'].apply(is_formatted)
data['has_attachment'] = data['email'].apply(has_attachment)
data['text_readability'] = data['email'].apply(text_readability)
data['sentiment_score'] = data['email'].apply(sentiment_score)
data['readability_score'] = data['email'].apply(readability_score)
#-----------------------------------------------

#-----------------------------------------------
# featuek es labelek szetszedese
X = data.drop(columns=['email', 'label', 'word_repetition'])  # szuksegtelen mezok dobasa
y = data['label']
#-----------------------------------------------
# Step 3: Traineles
#-----------------------------------------------
# adata szetszedese training and test setekbe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# INive Bayes model
nb_model = MultinomialNB()

# traineles
nb_model.fit(X_train, y_train)

# predikcio
y_pred = nb_model.predict(X_test)

# kiertekeles
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
#-----------------------------------------------
#-----------------------------------------------
# korelaciok kiszamitasa

# Calculate correlations
correlation_matrix = X.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# korelaciok es pontossag mentese
accuracy = accuracy_score(y_test, y_pred)
with open('model_performance.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy * 100:.2f}%\n')

# a matrix csv-be megy nem txt be
correlation_matrix.to_csv('correlation_matrix.csv')

# conoslera iras
print(f'Accuracy: {accuracy * 100:.2f}%')
print("Correlation Matrix:")
print(correlation_matrix)
#-----------------------------------------------