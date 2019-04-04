import csv

def clean_auction():
    """
    clean the data to store them into different files
        user: store the data of every user's record of auction ordered by time, every user has one file
        item: store the items and their highest price
    :return:
    """
    user = {}
    item = {}
    item_order = {}  # the dict used to sort
    with open('../resources/data/auction.csv') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            # store the item: id, time, price, item name
            if row['auctionid'] not in item:
                id = row['auctionid']
                item[id] = {}
                item[id]['id'] = row['auctionid']
                item[id]['time'] = float(row['bidtime'])
                item[id]['price'] = float(row['price'])
                item[id]['name'] = row['item']

                item_order[row['auctionid']] = float(row['price'])

            # store the user: price, time, item name
            if row['bidder'] not in user:
                user[row['bidder']] = {}
            index = len(user[row['bidder']])
            user[row['bidder']][index] = {}
            user[row['bidder']][index]['bidder'] = row['bidder']
            if "price" not in user[row['bidder']][index]:
                user[row['bidder']][index]['price'] = row['bid']
            else:
                user[row['bidder']][index]['price'] = max(user[row['bidder']][index]['price'], row['bid'])
            user[row['bidder']][index]['time'] = float(row['bidtime'])
            user[row['bidder']][index]['name'] = row['item']

    # write the item into file
    item_file = open("../resources/data/clean_data/item.txt", 'w')
    item_ordered = sorted(item_order.items(), key=lambda items: items[1], reverse=True)
    for v, d in item_ordered:
        item_file.write(
            item[v]['id'] + '\t' + str(item[v]['time']) + '\t' + str(item[v]['price']) + '\t' + item[v]['name'] + '\n')
    item_file.close()

    # write the user file
    for v in user:
        # sort the record of every user by time
        user_order = {}
        user_info = user[v]
        for value in user_info:
            user_order[value] = user_info[value]['time']

        user_ordered = sorted(user_order.items(), key=lambda items: items[1], reverse=True)
        bidder = v
        file_name = "../resources/data/clean_data/user/" + bidder + '_' + str(len(user[v])) + ".txt"
        f = open(file_name, 'w')
        for index, d in user_ordered:
            f.write(str(user[v][index]['price']) + '\t' + str(user[v][index]['time']) + '\t' + user[v][index][
                'name'] + '\n')
        f.close()

def clean_set():
    """
    clean data from training set
    :return:
    """
    file_prefix="../resources/data/"
    user={}
    item={}
    item_order={}  # used to sort the item by avg price
    with open(file_prefix+"TrainingSet.csv") as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            price=float(row['Price'])
            category=row['Category']  # the category of item
            person_id=row['PersonID']
            avg_price=float(row['AvgPrice'])

            # store the price of item
            if category not in item:
                item[category]={}
                item[category]['avg']=avg_price
                item[category]['prices']=[]
                item_order[category]=avg_price
            item[category]['prices'].append(price)

            # store the history of user
            if person_id not in user:
                user[person_id]={}

            # record the highest price if user has different willingness price for one item
            # we should record every price to regression
            if category in user[person_id]:
                user[person_id][category]['price']=max(user[person_id][category]['price'],price)
            else:
                user[person_id][category]={}
                user[person_id][category]['price']=price
                user[person_id][category]['avg']=avg_price
                user[person_id][category]['log']=[]
            user[person_id][category]['log'].append(price)

    # write the file
    write_file_prefix=file_prefix+"clean_data_2/"
    item_file=open(write_file_prefix+"item.txt",'w')
    # write item, sorted by avg price
    # format: category, avg price, history price(a list)
    item_ordered=sorted(item_order.items(), key=lambda items: items[1], reverse=True)
    for v,k in item_ordered:
        item_file.write(str(v)+'\t'+str(k))
        for p in item[v]['prices']:
            item_file.write('\t'+str(p))
        item_file.write('\n')
    item_file.close()

    # write history of user
    # format: category, max price, avg price, history price(a list)
    user_file_prefix=write_file_prefix+"/user/"
    for id in user:
        size=len(user[id])
        user_name=id+"_"+str(size)+".txt"
        user_file=open(user_file_prefix+user_name,'w')
        for category in user[id]:
            user_file.write(category+'\t'+str(user[id][category]['price'])+'\t'+str(user[id][category]['avg']))
            for p in user[id][category]['log']:
                user_file.write('\t'+str(p))
            user_file.write('\n')
        user_file.close()

if __name__ == '__main__':
    clean_set()