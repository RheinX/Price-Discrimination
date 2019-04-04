import numpy as np
import csv

if __name__ == '__main__':
    """
    clean the data to store them into different files
    user: store the data of every user's record of auction ordered by time, every user has one file
    item: store the items and their highest price
    """
    user={}
    item={}
    item_order={}   # the dict used to sort
    with open('../resources/data/auction.csv') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            # store the item: id, time, price, item name
            if row['auctionid'] not in item:
                id=row['auctionid']
                item[id] = {}
                item[id]['id']=row['auctionid']
                item[id]['time']=float(row['bidtime'])
                item[id]['price']=float(row['price'])
                item[id]['name']=row['item']

                item_order[row['auctionid']]=float(row['price'])


            # store the user: price, time, item name
            if row['bidder'] not in user:
                user[row['bidder']]={}
            index=len(user[row['bidder']])
            user[row['bidder']][index]={}
            user[row['bidder']][index]['bidder']=row['bidder']
            if "price" not in user[row['bidder']][index]:
                user[row['bidder']][index]['price'] = row['bid']
            else:
                user[row['bidder']][index]['price']=max(user[row['bidder']][index]['price'],row['bid'])
            user[row['bidder']][index]['time']=float(row['bidtime'])
            user[row['bidder']][index]['name']=row['item']

    # write the item into file
    item_file=open("../resources/data/clean_data/item.txt",'w')
    item_ordered=sorted(item_order.items(),key=lambda items:items[1],reverse=True)
    for v,d in item_ordered:
        item_file.write(item[v]['id']+'\t'+str(item[v]['time'])+'\t'+str(item[v]['price'])+'\t'+item[v]['name']+'\n')
    item_file.close()

    # write the user file
    for v in user:
        # sort the record of every user by time
        user_order={}
        user_info=user[v]
        for value in user_info:
            user_order[value]=user_info[value]['time']

        user_ordered=sorted(user_order.items(),key=lambda items:items[1], reverse=True)
        bidder = v
        file_name = "../resources/data/clean_data/user/" + bidder + '_' + str(len(user[v])) + ".txt"
        f = open(file_name, 'w')
        for index,d in user_ordered:
            f.write(str(user[v][index]['price'])+'\t'+str(user[v][index]['time'])+'\t'+user[v][index]['name']+'\n')
        f.close()