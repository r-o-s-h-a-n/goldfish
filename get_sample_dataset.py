import csv
import os
import datetime
import sys
import string

csv.field_size_limit(sys.maxsize)


PUBS = ('New York Times'
        , 'Breitbart'
        , 'Reuters'
        , 'CNN'
        , 'Washington Post'
        , 'Fox News'
        , 'Buzzfeed News'
        )


START_DATE = datetime.date(year = 2017, month = 1, day = 1)
END_DATE = datetime.date(year = 2017, month = 6, day = 1)

#######################################################
def strip_alpha(s):
    # removes alphabet characters from a string
    alpha = list((string.ascii_lowercase + string.ascii_uppercase).encode('utf-8'))
    out = ''
    for c in s:
        if c not in alpha:
            out += c
    return out.strip()



def format_date(date_str):
    # converts dates as strings to python dates
    # returns None if the date cannot be converted to a python date
    if ('/' not in date_str and '-' not in date_str):
        return None

    date_str = strip_alpha(date_str)
    date_str.strip('-')

    if '/' in date_str:
        date_spl = date_str.split('/')
    if '-' in date_str:
        date_spl = date_str.split('-')

    if len(date_spl[0]) > len(date_spl[2]):
        return datetime.date(int(date_spl[0]), int(date_spl[1]), int(date_spl[2]))

    if len(date_spl[2]) == 4:
        return datetime.date(int(date_spl[2]), int(date_spl[0]), int(date_spl[1]))

    try:
        return datetime.date(2000+int(date_spl[2]), int(date_spl[0]), int(date_spl[1]))
    except ValueError:
        print(date_str)
        return None

    return None


# compile revelant raw data files
fps = []
data_dir = os.path.join(os.getcwd(), 'data/kaggle_scrape/')
for fn in os.listdir(data_dir):
    if fn.endswith('.csv'):
        fps.append(os.path.join(data_dir, fn))

# get field names of data file
with open(fps[0], 'rU') as f:
    f_reader = csv.reader(f)
    field_names = next(f_reader)

# search for data rows that fit search criteria and write relevant rows to outfile
w = open('data/kaggle_scrape_big_sample.csv', 'w')
w_writer = csv.DictWriter(w, field_names)
w_writer.writeheader()

for fp in fps:
    with open(fp, 'rU') as  f:
        f_reader = csv.DictReader(f)

        for r in f_reader:

            pub_date = format_date(r['date'])
            if pub_date:

                # check conditions for adding row to outfile
                if r['publication'] in PUBS \
                    and pub_date >= START_DATE \
                    and pub_date <= END_DATE:

                    w_writer.writerow(r)

w.close()
