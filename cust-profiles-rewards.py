
# coding: utf-8

# In[ ]:


import pandas as pd
import json
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import time
from unidecode import unidecode
from itertools import chain
from multiprocessing import Pool


# In[ ]:


DATA_DIR = '/Users/ik/Data/qa/'


# In[ ]:


def timer(func):
    def wrapper(*args, **kwargs):
        t_start = time.time()
        res = func(*args, **kwargs)
        print("f: {} # elapsed time: {:.0f} m {:.0f}s".format(func.__name__.upper(), *divmod(time.time() - t_start, 60)))
        return res
    return wrapper


# In[ ]:


class RewardsDataProcessor(object):
    
    def __init__(self):
        
        # dictionary that maps CATEGORIES to QUESTION NUMBERS;
        # NOTE THE ABSENCE OF QUESTION 22!
        self.CAT_QUES = {"personal": [_ for _ in range(1,8)] + [25] + [_ for _ in range(70,74)],
                            "children": [_ for _ in range(11,22)] + [77,113,114],
                            "work": [5] + [_ for _ in range(27,32)],
                            "finance": [26] + [_ for _ in range(32,40)] + [115],
                            "insurance": [40,66,67,81] + [_ for _ in range(84,90)],
                            "transportation": [_ for _ in range(42,48)] + [91,92,116],
                            "phone": [_ for _ in range(48,54)] + [59,60,80],
                            "internet": [54,55,56,93],
                            "devices": [57,58,62,117],
                            "health": [63,64,65,118,119,120],
                            "shopping": [61,68,69,94,95],
                            "property": [8,9,10,41,90],
                            "media": [75,76] + [_ for _ in range(109,113)],
                            "opt-ins": [23,24,78,79,108],
                            "travel": [_ for _ in range(96,103)],
                            "interests": [74,82,83],
                            "drinks": [_ for _ in range(103,108)]}

        # reversed dictionary, question number -> category
        self._QUES_CAT = {q:k for k, v in self.CAT_QUES.items() for q in v}  

        # dictionary mapping QUESTION NUMBER to PROFILE ITEM
        self.QUES_ITEM = {1: "gender", 2: "dob", 3: "marital_status", 4: "home_postcode",
                        5: "work_postcode", 6: "lives_in_state", 7: "lives_in_area", 8: "people_in_household",
                        9: "housing_type", 10: "household_type", 11: "kids_u18_in_household",
                        12: "childs_gender_dob", 13: "childs_gender_dob", 14: "childs_gender_dob",
                        15: "childs_gender_dob", 16: "childs_gender_dob", 17: "childs_gender_dob",
                        18: "childs_gender_dob", 19: "childs_gender_dob", 20: "childs_gender_dob",
                        21: "childs_gender_dob",
                        22: "", 
                        23: "wants_be_in_focus_group", 24: "wants_phone_interview", 
                        25: "education", 26: "main_salary_earner", 27: "employment_status", 
                        28: "industry", 29: "occupation", 30: "company_size",
                        31: "company_annual_turnover", 32: "annual_income", 33: "annual_household_income",
                        34: "ways_to_pay_bills", 35: "financial_services", 36: "financial_institutions",
                        37: "main_financial_institutions", 38: "numb_credit_store_cards",
                        39: "total_credit_limit", 40: "insurance_policies", 41: "home_ownership_status",
                        42: "vehicle_owned", 43: "cond_most_used_vehicle_when_purchased",
                        44: "vehicle_makes_owned", 45: "value_most_used_vehicle_when_purchased",
                        46: "vehicle_types_owned", 47: "main_transport_to_work", 
                        48: "owns_mobile", 49: "mobile_brand", 50: "who_pays_mobile",
                        51: "mobile_network", 52: "mobile_on_contract", 53: "mobile_contract_expiration",
                        54: "internet_at_home", 55: "type_internet_at_home", 56: "isp",
                        57: "owns_computer", 58: "computer_type", 59: "landline_at_home", 
                        60: "landline_provider", 61: "online_purchasing_freq",
                        62: "owns_devices", 63: "smoker", 64: "wears_glasses_or_lenses",
                        65: "conditions_suffered", 66: "has_health_insurance", 
                        67: "private_health_insurance_with", 68: "role_in_buying_groceries",
                        69: "buying_groceries_online", 70: "born_in", 71: "ancestry",
                        72: "languages_at_home", 73: "religion", 74: "pets", 75: "pay_tv_at_home",
                        76: "pay_tv_provider", 77: "total_kids", 78: "okayed_kids_online_surveys",
                        79: "wants_sms_offers", 80: "mobile_number", 81: "vehicle_insurance_expiration",
                        82: "owns_swimming_pool", 83: "interested_in_activities", 
                        84: "home_building_insurance_expiration",
                        85: "home_contents_insurance_expiration", 86: "life_insurance_expiration",
                        87: "health_insurance_expiration", 88: "boat_insurance_expiration",
                        89: "caravan_insurance_expiration", 90: "bought_home_in",
                        91: "total_vehicles_in_household", 92: "year_bought_most_used_vehicle",
                        93: "social_networks", 94: "main_supermarkets_for_groceries", 
                        95: "regularly_shops_at_department_stopes", 96: "member_of_frequent_flyer",
                        97: "flights_past_12_months", 98: "purpose_flying_past_12_months",
                        99: "how_often_would_fly_for_business_a_year", 
                        100: "how_often_would_fly_for_leisure_a_year",
                        101: "on_holidays_goes_to", 102: "rented_a_car_past_12_months",
                        103: "regular_alcoholic_drinks", 104: "energy_drinks",
                        105: "sports_drinks", 106: "bottles_wine_a_month_at_household",
                        107: "how_much_ok_to_spend_bottle_wine", 108: "wants_wine_offers",
                        109: "reads_newspapers", 110: "reads_magazines", 111: "watches_sports",
                        112: "reads_news_portals", 113: "is_pregnant", 114: "baby_due",
                        115: "credit_card_types", 116: "plans_to_purchase_vehicle", 
                        117: "deviced_purchased_upgraded_past_12_month", 118: "type_of_cigarettes",
                        119: "brands_of_cigarettes", 120: "brands_of_cigarette_papers"}
        
    def get_month_year(self, st):
        """
        IN: a string expected to be a datetime like "Aug 14 2013  6:08PM"
        OUT: a string that only contains months and year extracted from the input string; format is '03/2016'
        """
    
        # empty st - return None straight away
        if not st:
            return None
    
        # is there are quotes, remove; create list of words in string
        wrd_lst = str(st).replace("'","").replace('"',"").split()
    
        # if st clearly doesn't look like "Aug 14 2013  6:08PM", return none
        if len(wrd_lst) != 4:
            return None
        else:
            try:
                """
                %b - Month as locale’s abbreviated name.
                %d - Day of the month as a zero-padded decimal number.
                %Y - Year with century as a decimal number.
                """
                extracted_date = datetime.strptime(" ".join(wrd_lst[:-1]), "%b %d %Y").strftime("%m/%Y") 
            except:
                # most likely in case the format happens to be wrong - then just return None
                extracted_date = None
    
        return extracted_date

    @timer
    def read_original_responses(self):
    
        # read responses, interpret everything as strings; NOTE: member pks will be strings too
        self._resp = pd.read_csv(DATA_DIR + "profileresponses.txt", sep="|", encoding='latin-1', dtype=str)
        self._resp["RespondedTime"] = self._resp["RespondedTime"].str.lower().apply(self.get_month_year)
        self._resp["ResponseDate"] = self._resp["ResponseDate"].apply(self.get_month_year)
    
        return self
    
    @timer
    def read_original_questions_answers_customers(self):
        
        self._questions = pd.read_csv(DATA_DIR + "Questions.txt", sep="|", dtype=str)
        
        self._answers = pd.read_csv(DATA_DIR + "Answers.txt", sep="|", dtype=str)
        self._answers["Descr"] = self._answers["Descr"].apply(lambda _: _.split(";")[-1].strip().lower())
        
        self._customers = pd.read_csv(DATA_DIR + "YC_member.txt", sep="|", encoding='latin-1', dtype=str)
        
        return self
    
    @timer
    def create_cust_name_id_dict(self):
        
        """
        create a dictionary of the sort {"member_pk": {"name": john, "last_name": "snow", "cust_id": 9465722},...}
        """
        self._names_by_mpk_dict = (self._customers.loc[:, ["member_pk", "FirstName", "LastName", "TicketekCustomerID"]].set_index("member_pk")
                                  .rename(columns={"FirstName": "name", "LastName": "last_name", "TicketekCustomerID": "cust_id"})
                                  .applymap(lambda _: unidecode(str(_).lower().strip()))
                                  .to_dict(orient='index'))
        
        return self
    
    @timer
    def merge_responses_and_names(self):
        
        self._rn = (self._resp.merge(self._answers.loc[:,["Answer_PK", "Question_PK", "Descr"]], 
                left_on=['Answer_PK',"Question_PK"], right_on=['Answer_PK',"Question_PK"], how='inner'))
#                  .merge(self._customers.loc[:,["member_pk","FirstName", "LastName", "TicketekCustomerID"]], 
#                         left_on="Member_PK", right_on="member_pk", how="inner"))

        self._rn["Item"] = self._rn["Question_PK"].apply(lambda x: self.QUES_ITEM[int(x)] if int(x) in self.QUES_ITEM else x)
        self._rn["Category"] = self._rn["Question_PK"].apply(lambda x: self._QUES_CAT[int(x)] if int(x) in self._QUES_CAT else x)
        
        self._rn.loc[:,"FullResponse"] = ((self._rn.loc[:,"ResponseText"].fillna('') + '&' + self._rn.loc[:,"ResponseDate"].fillna(''))
                                            .apply(lambda x: tuple(w if w else None for w in x.split("&"))))
        
        # where FullResponse is a tuple of two None, the real reply is in Descr
        bool_where_nones = self._rn.loc[:,"FullResponse"].apply(lambda _: all(w is None for w in _))
        self._rn.loc[bool_where_nones, "Reply"] = self._rn.loc[bool_where_nones,"Descr"]
        self._rn.loc[~bool_where_nones, "Reply"] = self._rn.loc[~bool_where_nones,"FullResponse"]
        
        return self
    
    @timer
    def split_df(self, df, PREF_CHUNK_SIZE=200000):
        
        """
        helper function that splits data frame df into chunks based on the
        preferred chunk size(in rows) PREF_CHUNK_SIZE
        """
        # how many unique member oks in df?
        #print(df.columns)
        n = len(df.Member_PK.unique())
        # so how many chunks will be needed? 
        chunks = n//PREF_CHUNK_SIZE + (n%PREF_CHUNK_SIZE > 0)
        # sorted list of all member pks
        srt_pks = sorted(list(set(df.Member_PK)))
        print("splitting data frame into {} chunks...".format(chunks))
        
        return [df.loc[df.Member_PK.isin(srt_pks[i*PREF_CHUNK_SIZE: (i+1)*PREF_CHUNK_SIZE]), :] 
                                                                                for i in range(chunks)]
    def series_to_tuple(self, ser):
        """
        IN: ser - a pandas series
        OUT: a tuple containing all elements of ser
        """
        return tuple([v if v else v for v in ser])
    
    def list_tuples_to_list_dicts(self, lst_tuples):
        return [{lst_tuples[0][i]: lst_tuples[1][i]} for i in range(len(lst_tuples[0]))]
    
    @timer
    def collect_responses(self, df, ll=['Member_PK', "Category", "Item"], al=["Reply", "RespondedTime"]):
    
        grouped_by_member_pk = df.groupby('Member_PK')  # Member_PK becomes index
        dict_list = []
    
        i = 0
        
        for mpk, d in grouped_by_member_pk:
        
            cust_dict = dict()   
            cust_data = d[["Category", "Item", "Reply", "RespondedTime"]].groupby(["Category", "Item"]).agg(self.series_to_tuple)
            """
            cust_data has Category and Item as indices:
                                                               Reply RespondedTime
                                            Category Item                         
                                            personal gender  (male,)    (10/2007,)
            """
            for row in cust_data.iterrows():
                
                """
                row[0] is a tuple of (category_name, item_name)
                row[1] is a series with named entries
                
                -- case 1:
                
                row[1][Reply] is a tuple like 
                            (coles, aldi, woolworths, iga / supa iga)
                while row[1]["RespondedTime"] is another tuple with matching times:
                             (12/2011, 12/2011, 12/2011)
                             
                -- case 2:
                
                row[1][Reply] can be like
                            ((None, 01/2003),)
                and row[1]["RespondedTime"] is still (12/2011,)
                
                -- case 3 (children details):
                
                row[1][Reply] is ((female, 01/2005), (female, 01/2007))
                and row[1]["RespondedTime"] is (10/2007, 10/2007)
                
                """
                category_name, item_name = row[0]
                
                
                if isinstance(row[1]["Reply"][0], tuple):
                    if all(row[1]["Reply"][0]):
                        r = {tp[0]: tp[1] for tp in row[1]["Reply"]}
                    else:
                        r = {[tuple_part for tuple_part in row[1]["Reply"][0] if tuple_part][0]: row[1]["RespondedTime"][0]}
                else:
                    r = {row[1]["Reply"][i]: row[1]["RespondedTime"][i] for i in range(len(row[1]["Reply"]))}
                
                if mpk in cust_dict:
                    
                    if category_name in cust_dict[mpk]:
                        cust_dict[mpk][category_name].update({item_name: r})
                    else:
                        cust_dict[mpk].update({category_name: {item_name: r}})
                else:
                    cust_dict.update({mpk: {category_name: {item_name: r}}})
                
                if mpk in self._names_by_mpk_dict:
                    cust_dict[mpk].update(self._names_by_mpk_dict[mpk])
                    
                
        
            dict_list.append(cust_dict)
            
            i += 1
            
            if i % 5000 == 0:
                print("processed {} customers".format(i))
    
        return dict_list
    
    @timer
    def collect_responses_parallel(self):
        
        self.rsps = []
        
        df_split = self.split_df(self._rn, PREF_CHUNK_SIZE=10000)  # [df1, df2, ..]
        
        npairs = len(df_split)//2
        
        for i in range(npairs):
            pool = Pool(2)  
            print("made pool")
            lst_dicts = pool.map(self.collect_responses, df_split[i*2:(i+1)*2])
            print("did map")
            pool.close()
            pool.join()
            
            print("extending list")
            self.rsps.extend([d for l in lst_dicts for d in l])
            print("dicts collected so far: {}".format(len(self.rsps)))
        
        return self


# In[ ]:


if __name__ == '__main__':
    
    rp = (RewardsDataProcessor()
            .read_original_responses()
            .read_original_questions_answers_customers()
            .create_cust_name_id_dict()
            .merge_responses_and_names())
    
    rsp = rp.collect_responses(rp._rn)


# In[ ]:


json.dump(rsp, open('rewards-qa-data-18102017.json', 'w'))

