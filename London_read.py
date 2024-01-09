import preprocessor as p
import xlrd
import csv

p.set_options(p.OPT.MENTION, p.OPT.URL, p.OPT.RESERVED, p.OPT.EMOJI)
#wb = xlrd.open_workbook("USA/coverage.xlsx")
#wb = xlrd.open_workbook("USA/dashboard.xlsx")
wb = xlrd.open_workbook("Stream.xlsx")
sheet = wb.sheet_by_index(0)
#skip_val = sheet.cell_value(-1, 1)
count = 0
sheet.cell_value(0, 0)
#for i in range(sheet.ncols):
    #print(sheet.cell_value(0, i))
tweets = open("tweets.csv", "w")
writer = csv.writer(tweets)
for i in range(sheet.nrows):
    #if 'Los Angeles, CA' in sheet.cell_value(i, 12):

    #Filtering tweets only from the London area based on latitude and longitude
    '''lati = sheet.cell_value(i, 9)
    longi = sheet.cell_value(i, 10)
    f = float(lati)
    print(f)
    print(float(lati), float(longi))'''
    lati = sheet.cell_value(i, 9)
    longi = sheet.cell_value(i, 10)
    #if 51.3794444 <= lati <= 51.6275813 and -0.4598808 <= longi <= 0.0994444:
    if 'London' in sheet.cell_value(i, 12):
        #Want to keep only tweets in the English language
        if 'en' in sheet.cell_value(i, 17):
            #count += 1
            #print(sheet.cell_value(i, 12))
            line = sheet.cell_value(i, 6)
            line = line.encode('ascii', 'ignore').decode('ascii')
            line = p.clean(line)
            writer.writerow([line,lati,longi])

tweets.close()
