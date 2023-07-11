#!/usr/bin/env python
# coding: utf-8

# In[25]:


from pyhive import hive
import pandas as pd
import numpy as np
import re
import py2neo
import subprocess
import logging
import datetime
pd.options.display.max_colwidth=2000
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
import time
import itertools
from sklearn.preprocessing import MinMaxScaler
import joblib
import multiprocessing
import os
import logging
import math


# In[26]:


# #Setting up logging mech.
#logging.basicConfig(level=logging.INFO,filename='Network_scoring_log'+datetime.datetime.now().strftime("%d%h%y")+'.log',filemode='w', format='%(name)s - %(levelname)s - %(message)s')


# #### Connect to Hive

# In[27]:


#conn=hive.Connection(host="10.0.0.96", port=10000, username="hive") 
conn=hive.Connection(host="10.0.9.51", port=10000, username="hive") 


# #### Read Leads GSTINs

# In[28]:


#df_leads = pd.read_sql("select supplier_gstin as gstin_id from genome_distribution.network_gn_leads_1_mth", conn)


# In[29]:


#remove the gstins which take very long time to extract the paths and the code gets stuck
#gstin_list_not_to_be_taken=['27GIDPP2672A1ZC','34NLEPS9553N1ZZ','33AAFCI2906R1ZA','17ARTPP3068P1ZQ','29AAACW0315K1ZD',
#                           '29DZDPK6396A1Z5','09ALGPS8092P1ZZ']
#df_leads=df_leads.drop([x for x in gstin_list_not_to_be_taken if x in df_leads.gstin_id],axis=1)


# In[30]:


#print("total shpe of dataframe(leads)",df_leads.shape)
#print("unique gstins",df_leads.nunique())
#df_leads=df_leads.drop_duplicates(keep='first')
#print("final shape after removeing duplicates",df_leads.shape)


# In[32]:


#run the code the loop(count of times the loop should run)
#count_of_df=math.ceil(df_leads.shape[0]/1000)
#print("count of times loop should run",count_of_df)


# #### Read path features
# 

# In[36]:


df_path_feature=pd.read_sql('select gstin_id as gstin_id,reg_type_sc as reg_type_sc,cstn_bus_sc_g1 as cstn_bus_sc_G1,cstn_bus_sc_g2 as cstn_bus_sc_G2,cstn_bus_sc_oth as cstn_bus_sc_OTH,state_North as state_North,state_North_East as state_North_East,state_South as state_South,state_West as state_West,state_East as state_East,state_UT as state_UT,ntr_pos_LES as ntr_pos_LES,ntr_pos_sc_OWN as ntr_pos_sc_OWN,ntr_pos_sc_CON as ntr_pos_sc_CON,ntr_pos_sc_OTH as ntr_pos_sc_OTH,is_migrated as is_migrated,age_of_gstin as age_of_gstin,gti_range as gti_range,aadhar_linked as aadhar_linked,overall_score as overall_score,isnil as isnil,r3b_sup_details_osup_nil_exmp_txval as r3b_sup_details_osup_nil_exmp_txval,r3b_sup_details_osup_nongst_txval as r3b_sup_details_osup_nongst_txval,r3b_taxable_turnover as r3b_taxable_turnover,r3b_itc_availed as r3b_itc_availed,lgr_cash_utilized_3b as lgr_cash_utilized_3b,r1_elig_vs_r1_filed as r1_elig_vs_r1_filed,r3b_elig_vs_r3b_filed as r3b_elig_vs_r3b_filed,r3b_sup_details_osup_det_tax as r3b_sup_details_osup_det_tax,r3b_sup_details_osup_zero_tax as r3b_sup_details_osup_zero_tax,R3B_4A_1_2_3_4 as R3B_4A_1_2_3_4,extra_txval as extra_txval,cash_vs_total as cash_vs_total,supply_loss as supply_loss,b2b_vs_total_tax_ratio as b2b_vs_total_tax_ratio,b2cl_vs_total_tax_ratio as b2cl_vs_total_tax_ratio, pagerank as pagerank,degree as degree from genome_distribution.df_final_path_score3',conn)
df_path_feature['is_migrated']=df_path_feature.is_migrated.astype('float64')


# #### Read GSTIN features for Network score

# In[37]:


data_network_score = pd.read_sql("select unique_gstin as gstin_id,itc_passed_cnt ,itc_utilized_cnt ,missing_tax_cnt ,high_itc_cnt ,hi_b2c_cnt ,suspicious_tran_cnt ,age_of_gstin as age_of_gstin_nw,suo_flag ,cancel_flag ,exporter_flag ,cash_vs_total as cash_vs_total_nw,cnl_ewb_per ,cnt_sensitive_hsn  from genome_distribution.network_table_network_score",conn)
data_network_score['age_of_gstin_nw'].fillna(data_network_score['age_of_gstin_nw'].mean(),inplace=True)
sc = MinMaxScaler()
sc.fit(data_network_score[['age_of_gstin_nw']])
data_network_score['age_of_gstin_nw']=sc.transform(data_network_score[['age_of_gstin_nw']])


# #### Load LightGBM Model pkl file

# In[38]:


best_model = joblib.load('/home/such4579/Network_scoring_uat_code/786323/New_LGBM_Model.pkl')


# ### Functions
# 

# #### Counter Fn to aid identifying a path from Neo4j result

# In[39]:


def incr_path(x):
    global cnt
    
    if x==0:
        cnt=cnt+1
        return cnt
    else:
        return cnt


# #### Extract Path from Neo4J

# In[40]:


def get_all_path_new(gstin,h,error_paths_ns):
    error_df=pd.DataFrame()
    df_all=pd.DataFrame()
    try:
        df_path_data_fwd=pd.DataFrame()
        df_path_data_bwd=pd.DataFrame()
        df_path_data_bwd_final=pd.DataFrame()
        df_path_data_fwd_final=pd.DataFrame()
    
        global cnt 
        h = py2neo.Graph('http://10.0.2.90:7474',auth =('neo4j','neo4jPROD'))
        query = h.run("""Match (g:GSTIN{gstin_id:$x}) 
            Call gstn.networkScoreFwd(g,10,0, 'Rolling_12m', true, false) 
            Yield paths unwind paths as p UNWIND nodes(p) as node 
            WITH collect(node.gstin_id) as names,collect(node.depth) as depth RETURN names,depth
            UNION
            Match (g:GSTIN {gstin_id:$x}) 
            Call gstn.networkScoreBwd(g,10,0, 'Rolling_12m', true, false) 
            Yield paths unwind paths as p     
            UNWIND nodes(p) as node WITH collect(node.gstin_id) as names,collect(node.depth) as depth 
            RETURN names,depth""",x=gstin).data()
      
        cnt=-1
        if len(query[0]['names']) > 1 or len(query[1]['names']) > 1:
            df_path_data_fwd['gstin_id']=query[0]['names']
            df_path_data_fwd['depth']=query[0]['depth']
            df_path_data_fwd['index_new']=df_path_data_fwd['depth'].apply(lambda x : incr_path(x))
            df_path_data_fwd_final=pd.DataFrame(df_path_data_fwd.groupby('index_new')['gstin_id'].apply(list))
            df_path_data_fwd_final['root_gstin']=gstin
            df_path_data_fwd_final=df_path_data_fwd_final.loc[df_path_data_fwd_final['gstin_id'].str.len()>1]
            df_path_data_fwd_final=df_path_data_fwd_final.reset_index()
       
        
            df_path_data_bwd['gstin_id']=query[1]['names']
            df_path_data_bwd['depth']=query[1]['depth']
            df_path_data_bwd['index_new']=df_path_data_bwd['depth'].apply(lambda x : incr_path(x))
            df_path_data_bwd.depth=-(df_path_data_bwd.depth)
            df_path_data_bwd_final=pd.DataFrame(df_path_data_bwd.groupby('index_new')['gstin_id'].apply(list))
            df_path_data_bwd_final=df_path_data_bwd_final.loc[df_path_data_bwd_final['gstin_id'].str.len()>1]
            
            df_path_data_fwd_final['k']=1
            df_path_data_bwd_final['k']=1
        if df_path_data_fwd_final.shape[0]!=0 and df_path_data_bwd_final.shape[0]!=0:
        
            df_all=pd.merge(df_path_data_fwd_final, df_path_data_bwd_final, on='k').drop('k',1)
            df_all['gstin_id']=df_all['gstin_id_x']+df_all['gstin_id_y']
            df_all.drop(['gstin_id_x','gstin_id_y'], axis=1,inplace=True)
        
        elif df_path_data_fwd_final.shape[0]==0 and df_path_data_bwd_final.shape[0]==0:
            df_all=pd.DataFrame()
        elif df_path_data_fwd_final.shape[0]==0:
            df_path_data_bwd_final['root_gstin']=gstin
            df_all=df_path_data_bwd_final.reset_index()
        else:
            df_all=df_path_data_fwd_final
    except Exception as e:
        error={}
        error['gstin_id']=g
        error['error']= e
        error_df=error_df.append(error,ignore_index=True)
        error_paths_ns.df=error_df
    return df_all 


# #### Multiprocess Function to fetch the paths

# In[41]:


def multiprocess_func(gstin_path):
    manager=multiprocessing.Manager()
  
    final_master=pd.DataFrame()
    no_paths_final_df=pd.DataFrame()
    error_final_df=pd.DataFrame()
    error_paths_df=pd.DataFrame()

    l =len(gstin_path)
    r= l+(20-(l%20))
    #h1 = py2neo.Graph('http://10.0.7.22:7474',auth =('neo4j','neo4juat'))
    h1 = py2neo.Graph('http://10.0.2.90:7474',auth =('neo4j','neo4jPROD'))

    for i in range(0,r,20):
        print(i)
        logging.info(i)
        if (l-i)<20:
            gstin_lst=gstin_path[i:]
        else:
            gstin_lst=gstin_path[i:i+20]
        leads_gstin=np.array_split(gstin_lst,20)
        
        #print(leads_gstin[0])
        
        master_df1=pd.DataFrame()
        master1=manager.Namespace()
        master1.df=master_df1

        master_df2=pd.DataFrame()
        master2=manager.Namespace()
        master2.df=master_df2

        master_df3=pd.DataFrame()
        master3=manager.Namespace()
        master3.df=master_df3

        master_df4=pd.DataFrame()
        master4=manager.Namespace()
        master4.df=master_df4

        master_df5=pd.DataFrame()
        master5=manager.Namespace()
        master5.df=master_df5

        master_df6=pd.DataFrame()
        master6=manager.Namespace()
        master6.df=master_df6

        master_df7=pd.DataFrame()
        master7=manager.Namespace()
        master7.df=master_df7

        master_df8=pd.DataFrame()
        master8=manager.Namespace()
        master8.df=master_df8

        master_df9=pd.DataFrame()
        master9=manager.Namespace()
        master9.df=master_df9

        master_df10=pd.DataFrame()
        master10=manager.Namespace()
        master10.df=master_df10

        master_df11=pd.DataFrame()
        master11=manager.Namespace()
        master11.df=master_df11

        master_df12=pd.DataFrame()
        master12=manager.Namespace()
        master12.df=master_df12

        master_df13=pd.DataFrame()
        master13=manager.Namespace()
        master13.df=master_df13

        master_df14=pd.DataFrame()
        master14=manager.Namespace()
        master14.df=master_df14

        master_df15=pd.DataFrame()
        master15=manager.Namespace()
        master15.df=master_df15

        master_df16=pd.DataFrame()
        master16=manager.Namespace()
        master16.df=master_df16

        master_df17=pd.DataFrame()
        master17=manager.Namespace()
        master17.df=master_df17

        master_df18=pd.DataFrame()
        master18=manager.Namespace()
        master18.df=master_df18

        master_df19=pd.DataFrame()
        master19=manager.Namespace()
        master19.df=master_df19

        master_df20=pd.DataFrame()
        master20=manager.Namespace()
        master20.df=master_df20

        no_path_df1=pd.DataFrame()
        no_path_ns1=manager.Namespace()
        no_path_ns1.df=no_path_df1

        no_path_df2=pd.DataFrame()
        no_path_ns2=manager.Namespace()
        no_path_ns2.df=no_path_df2

        no_path_df3=pd.DataFrame()
        no_path_ns3=manager.Namespace()
        no_path_ns3.df=no_path_df3

        no_path_df4=pd.DataFrame()
        no_path_ns4=manager.Namespace()
        no_path_ns4.df=no_path_df4

        no_path_df5=pd.DataFrame()
        no_path_ns5=manager.Namespace()
        no_path_ns5.df=no_path_df5

        no_path_df6=pd.DataFrame()
        no_path_ns6=manager.Namespace()
        no_path_ns6.df=no_path_df6

        no_path_df7=pd.DataFrame()
        no_path_ns7=manager.Namespace()
        no_path_ns7.df=no_path_df7

        no_path_df8=pd.DataFrame()
        no_path_ns8=manager.Namespace()
        no_path_ns8.df=no_path_df8

        no_path_df9=pd.DataFrame()
        no_path_ns9=manager.Namespace()
        no_path_ns9.df=no_path_df9

        no_path_df10=pd.DataFrame()
        no_path_ns10=manager.Namespace()
        no_path_ns10.df=no_path_df10

        no_path_df11=pd.DataFrame()
        no_path_ns11=manager.Namespace()
        no_path_ns11.df=no_path_df11
        no_path_df12=pd.DataFrame()
        no_path_ns12=manager.Namespace()
        no_path_ns12.df=no_path_df12

        no_path_df13=pd.DataFrame()
        no_path_ns13=manager.Namespace()
        no_path_ns13.df=no_path_df13

        no_path_df14=pd.DataFrame()
        no_path_ns14=manager.Namespace()
        no_path_ns14.df=no_path_df14

        no_path_df15=pd.DataFrame()
        no_path_ns15=manager.Namespace()
        no_path_ns15.df=no_path_df15

        no_path_df16=pd.DataFrame()
        no_path_ns16=manager.Namespace()
        no_path_ns16.df=no_path_df16

        no_path_df17=pd.DataFrame()
        no_path_ns17=manager.Namespace()
        no_path_ns17.df=no_path_df17

        no_path_df18=pd.DataFrame()
        no_path_ns18=manager.Namespace()
        no_path_ns18.df=no_path_df18

        no_path_df19=pd.DataFrame()
        no_path_ns19=manager.Namespace()
        no_path_ns19.df=no_path_df19

        no_path_df20=pd.DataFrame()
        no_path_ns20=manager.Namespace()
        no_path_ns20.df=no_path_df20




        error_df1=pd.DataFrame()
        error_ns1=manager.Namespace()
        error_ns1.df=error_df1

        error_df2=pd.DataFrame()
        error_ns2=manager.Namespace()
        error_ns2.df=error_df2

        error_df3=pd.DataFrame()
        error_ns3=manager.Namespace()
        error_ns3.df=error_df3

        error_df4=pd.DataFrame()
        error_ns4=manager.Namespace()
        error_ns4.df=error_df4

        error_df5=pd.DataFrame()
        error_ns5=manager.Namespace()
        error_ns5.df=error_df5

        error_df6=pd.DataFrame()
        error_ns6=manager.Namespace()
        error_ns6.df=error_df6

        error_df7=pd.DataFrame()
        error_ns7=manager.Namespace()
        error_ns7.df=error_df7

        error_df8=pd.DataFrame()
        error_ns8=manager.Namespace()
        error_ns8.df=error_df8

        error_df9=pd.DataFrame()
        error_ns9=manager.Namespace()
        error_ns9.df=error_df9

        error_df10=pd.DataFrame()
        error_ns10=manager.Namespace()
        error_ns10.df=error_df10

        error_df11=pd.DataFrame()
        error_ns11=manager.Namespace()
        error_ns11.df=error_df11

        error_df12=pd.DataFrame()
        error_ns12=manager.Namespace()
        error_ns12.df=error_df12

        error_df13=pd.DataFrame()
        error_ns13=manager.Namespace()
        error_ns13.df=error_df13

        error_df14=pd.DataFrame()
        error_ns14=manager.Namespace()
        error_ns14.df=error_df14

        error_df15=pd.DataFrame()
        error_ns15=manager.Namespace()
        error_ns15.df=error_df15

        error_df16=pd.DataFrame()
        error_ns16=manager.Namespace()
        error_ns16.df=error_df16

        error_df17=pd.DataFrame()
        error_ns17=manager.Namespace()
        error_ns17.df=error_df17

        error_df18=pd.DataFrame()
        error_ns18=manager.Namespace()
        error_ns18.df=error_df18

        error_df19=pd.DataFrame()
        error_ns19=manager.Namespace()
        error_ns19.df=error_df19

        error_df20=pd.DataFrame()
        error_ns20=manager.Namespace()
        error_ns20.df=error_df20

        error_paths_df1=pd.DataFrame()
        error_paths_ns1=manager.Namespace()
        error_paths_ns1.df=error_paths_df1

        error_paths_df2=pd.DataFrame()
        error_paths_ns2=manager.Namespace()
        error_paths_ns2.df=error_paths_df2

        error_paths_df3=pd.DataFrame()
        error_paths_ns3=manager.Namespace()
        error_paths_ns3.df=error_paths_df3

        error_paths_df4=pd.DataFrame()
        error_paths_ns4=manager.Namespace()
        error_paths_ns4.df=error_paths_df4

        error_paths_df5=pd.DataFrame()
        error_paths_ns5=manager.Namespace()
        error_paths_ns5.df=error_paths_df5


        error_paths_df6=pd.DataFrame()
        error_paths_ns6=manager.Namespace()
        error_paths_ns6.df=error_paths_df6

        error_paths_df7=pd.DataFrame()
        error_paths_ns7=manager.Namespace()
        error_paths_ns7.df=error_paths_df7

        error_paths_df8=pd.DataFrame()
        error_paths_ns8=manager.Namespace()
        error_paths_ns8.df=error_paths_df8

        error_paths_df9=pd.DataFrame()
        error_paths_ns9=manager.Namespace()
        error_paths_ns9.df=error_paths_df9

        error_paths_df10=pd.DataFrame()
        error_paths_ns10=manager.Namespace()
        error_paths_ns10.df=error_paths_df10

        error_paths_df11=pd.DataFrame()
        error_paths_ns11=manager.Namespace()
        error_paths_ns11.df=error_paths_df11

        error_paths_df12=pd.DataFrame()
        error_paths_ns12=manager.Namespace()
        error_paths_ns12.df=error_paths_df12

        error_paths_df13=pd.DataFrame()
        error_paths_ns13=manager.Namespace()
        error_paths_ns13.df=error_paths_df13

        error_paths_df14=pd.DataFrame()
        error_paths_ns14=manager.Namespace()
        error_paths_ns14.df=error_paths_df14

        error_paths_df15=pd.DataFrame()
        error_paths_ns15=manager.Namespace()
        error_paths_ns15.df=error_paths_df15

        error_paths_df16=pd.DataFrame()
        error_paths_ns16=manager.Namespace()
        error_paths_ns16.df=error_paths_df16

        error_paths_df17=pd.DataFrame()
        error_paths_ns17=manager.Namespace()
        error_paths_ns17.df=error_paths_df17

        error_paths_df18=pd.DataFrame()
        error_paths_ns18=manager.Namespace()
        error_paths_ns18.df=error_paths_df18

        error_paths_df19=pd.DataFrame()
        error_paths_ns19=manager.Namespace()
        error_paths_ns19.df=error_paths_df19

        error_paths_df20=pd.DataFrame()
        error_paths_ns20=manager.Namespace()
        error_paths_ns20.df=error_paths_df20

        p1 = multiprocessing.Process(target=main, args=(leads_gstin[0],master1,no_path_ns1,error_ns1,error_paths_ns1,h1))

        p2 = multiprocessing.Process(target=main, args=(leads_gstin[1],master2,no_path_ns2,error_ns2,error_paths_ns2,h1))
        p3 = multiprocessing.Process(target=main, args=(leads_gstin[2],master3,no_path_ns3,error_ns3,error_paths_ns3,h1))
        p4 = multiprocessing.Process(target=main, args=(leads_gstin[3],master4,no_path_ns4,error_ns4,error_paths_ns4,h1))
        p5 = multiprocessing.Process(target=main, args=(leads_gstin[4],master5,no_path_ns5,error_ns5,error_paths_ns5,h1))

        p6 = multiprocessing.Process(target=main, args=(leads_gstin[5],master6,no_path_ns6,error_ns6,error_paths_ns6,h1))
        p7 = multiprocessing.Process(target=main, args=(leads_gstin[6],master7,no_path_ns7,error_ns7,error_paths_ns7,h1))
        p8 = multiprocessing.Process(target=main, args=(leads_gstin[7],master8,no_path_ns8,error_ns8,error_paths_ns8,h1))
        p9 = multiprocessing.Process(target=main, args=(leads_gstin[8],master9,no_path_ns9,error_ns9,error_paths_ns9,h1))
        p10 = multiprocessing.Process(target=main, args=(leads_gstin[9],master10,no_path_ns10,error_ns10,error_paths_ns10,h1))

        p11 = multiprocessing.Process(target=main, args=(leads_gstin[10],master11,no_path_ns11,error_ns11,error_paths_ns11,h1))
        p12 = multiprocessing.Process(target=main, args=(leads_gstin[11],master12,no_path_ns12,error_ns12,error_paths_ns12,h1))
        p13 = multiprocessing.Process(target=main, args=(leads_gstin[12],master13,no_path_ns13,error_ns13,error_paths_ns13,h1))
        p14 = multiprocessing.Process(target=main, args=(leads_gstin[13],master14,no_path_ns14,error_ns14,error_paths_ns14,h1))
        p15 = multiprocessing.Process(target=main, args=(leads_gstin[14],master15,no_path_ns15,error_ns15,error_paths_ns15,h1))

        p16 = multiprocessing.Process(target=main, args=(leads_gstin[15],master16,no_path_ns16,error_ns16,error_paths_ns16,h1))
        p17 = multiprocessing.Process(target=main, args=(leads_gstin[16],master17,no_path_ns17,error_ns17,error_paths_ns17,h1))
        p18 = multiprocessing.Process(target=main, args=(leads_gstin[17],master18,no_path_ns18,error_ns18,error_paths_ns18,h1))
        p19 = multiprocessing.Process(target=main, args=(leads_gstin[18],master19,no_path_ns19,error_ns19,error_paths_ns19,h1))
        p20 = multiprocessing.Process(target=main, args=(leads_gstin[19],master20,no_path_ns20,error_ns20,error_paths_ns20,h1))
     
        p1.start()

        p2.start()
        p3.start()
        p4.start()
        p5.start()
    
        p6.start()
        p7.start()
        p8.start()
        p9.start()
        p10.start()

        p11.start()
        p12.start()
        p13.start()
        p14.start()
        p15.start()

        p16.start()
        p17.start()
        p18.start()
        p19.start()
        p20.start()
   
   
        p1.join()

        p2.join()
        p3.join()
        p4.join()
        p5.join()

        p6.join()
        p7.join()
        p8.join()
        p9.join()
        p10.join()

        p11.join()
        p12.join()
        p13.join()
        p14.join()
        p15.join()

        p16.join()
        p17.join()
        p18.join()
        p19.join()
        p20.join()
    
        final_master_t=pd.concat([master1.df,master2.df,master3.df,master4.df,master5.df,master6.df,master7.df,master8.df,
                                 master9.df,master10.df, master11.df,master12.df,master13.df,master14.df,master15.df,
                                  master16.df,master17.df,master18.df,master19.df,master20.df],sort=True)

        no_paths_final_df_t=pd.concat([no_path_ns1.df,no_path_ns2.df,no_path_ns3.df,no_path_ns4.df,no_path_ns5.df,
                                       no_path_ns6.df,no_path_ns7.df,no_path_ns8.df,no_path_ns9.df,no_path_ns10.df,
                                      no_path_ns11.df,no_path_ns12.df,no_path_ns13.df,no_path_ns14.df,no_path_ns15.df,
                                       no_path_ns16.df,no_path_ns17.df,no_path_ns18.df,no_path_ns19.df,no_path_ns20.df],sort=True)
        error_final_df_t=pd.concat([error_ns1.df,error_ns2.df,error_ns3.df,error_ns4.df,error_ns5.df,
                                       error_ns6.df,error_ns7.df,error_ns8.df,error_ns9.df,error_ns10.df,
                                      error_ns11.df,error_ns12.df,error_ns13.df,error_ns14.df,error_ns15.df,
                                       error_ns16.df,error_ns17.df,error_ns18.df,error_ns19.df,error_ns20.df],sort=True)
        error_paths_df_t=pd.concat([error_paths_ns1.df,error_paths_ns2.df,error_paths_ns3.df,error_paths_ns4.df,error_paths_ns5.df,
                                       error_paths_ns6.df,error_paths_ns7.df,error_paths_ns8.df,error_paths_ns9.df,error_paths_ns10.df,
                                      error_paths_ns11.df,error_paths_ns12.df,error_paths_ns13.df,error_paths_ns14.df,error_paths_ns15.df,
                                       error_paths_ns16.df,error_paths_ns17.df,error_paths_ns18.df,error_paths_ns19.df,error_paths_ns20.df],sort=True)
        final_master=final_master.append(final_master_t,ignore_index=True)
        final_master1=pd.DataFrame(final_master)
        #print("paths",final_master1.head(30))
        final_master['tmp_st']=final_master['gstin_id'].apply(lambda x : '|'.join(map(str,x)))
        final_master.drop_duplicates(subset=['tmp_st'], keep='first',inplace=True)
        no_paths_final_df=no_paths_final_df.append(no_paths_final_df_t,ignore_index=True)
        error_final_df=error_final_df.append(error_final_df_t,ignore_index=True)
        error_paths_df=error_paths_df.append(error_paths_df_t,ignore_index=True)
    print("complete fetching paths")
    logging.info("complete fetching paths")
    return final_master, no_paths_final_df, error_final_df,error_paths_df


# #### Feature Engineering

# In[42]:


def feature_extraction(final_master,namespace):
    
    final_master['state_cnt']=final_master.apply(lambda x: (len(list(filter(re.compile("^{}".format(x.root_gstin[0:2])).match,x['unique_gstin'])))-1)/(len(x['unique_gstin'])-1),axis=1)
    fn1=['reg_type_sc', 'cstn_bus_sc_g1', 'cstn_bus_sc_g2', 'cstn_bus_sc_oth', 'state_north', 'state_north_east', 
         'state_south', 'state_west', 'state_east', 'state_ut', 'ntr_pos_les', 'ntr_pos_sc_own', 'ntr_pos_sc_con', 
         'ntr_pos_sc_oth', 'is_migrated', 'age_of_gstin', 'gti_range', 'aadhar_linked', 'overall_score', 'isnil', 
         'r3b_sup_details_osup_nil_exmp_txval', 'r3b_sup_details_osup_nongst_txval', 'r3b_taxable_turnover', 
         'r3b_itc_availed', 'lgr_cash_utilized_3b', 'r1_elig_vs_r1_filed', 'r3b_elig_vs_r3b_filed', 
         'r3b_sup_details_osup_det_tax', 'r3b_sup_details_osup_zero_tax', 'r3b_4a_1_2_3_4', 
         'extra_txval', 'cash_vs_total', 'supply_loss', 'b2b_vs_total_tax_ratio', 'b2cl_vs_total_tax_ratio','pagerank',
         'degree']
    print("state cnt done")
    logging.info("state cnt done")
    nw1=[ 'itc_passed_cnt', 'itc_utilized_cnt', 'missing_tax_cnt', 'high_itc_cnt', 'hi_b2c_cnt', 'suspicious_tran_cnt',
          'age_of_gstin_nw', 'suo_flag', 'cancel_flag', 'exporter_flag', 'cash_vs_total_nw', 'cnl_ewb_per', 
         'cnt_sensitive_hsn']
    final_master[nw1]=final_master['unique_gstin'].apply(lambda x : ( df_network.loc[df_network.gstin_id.isin(x)][nw1].mean()))
    print("network features extracted")
    logging.info("network features extracted")
    final_master[fn1]=final_master['unique_gstin'].apply(lambda x : ( df_path_ft.loc[df_path_ft.gstin_id.isin(x)][fn1].mean()))
    final_master=final_master[['root_gstin',  'unique_gstin', 'state_cnt', 'itc_passed_cnt', 'itc_utilized_cnt', 
                              'missing_tax_cnt', 'high_itc_cnt', 'hi_b2c_cnt', 'suspicious_tran_cnt', 'age_of_gstin_nw',
                              'suo_flag', 'cancel_flag', 'exporter_flag', 'cash_vs_total_nw', 'cnl_ewb_per',
                              'cnt_sensitive_hsn', 'reg_type_sc', 'cstn_bus_sc_g1', 'cstn_bus_sc_g2', 'cstn_bus_sc_oth',
                              'state_north', 'state_north_east', 'state_south', 'state_west', 'state_east', 'state_ut',
                              'ntr_pos_les', 'ntr_pos_sc_own', 'ntr_pos_sc_con', 'ntr_pos_sc_oth', 'is_migrated', 
                              'age_of_gstin', 'gti_range', 'aadhar_linked', 'overall_score', 'isnil', 
                              'r3b_sup_details_osup_nil_exmp_txval', 'r3b_sup_details_osup_nongst_txval', 
                              'r3b_taxable_turnover', 'r3b_itc_availed', 'lgr_cash_utilized_3b', 'r1_elig_vs_r1_filed', 
                              'r3b_elig_vs_r3b_filed', 'r3b_sup_details_osup_det_tax', 'r3b_sup_details_osup_zero_tax',
                              'r3b_4a_1_2_3_4', 'extra_txval', 'cash_vs_total', 'supply_loss', 'b2b_vs_total_tax_ratio',
                              'b2cl_vs_total_tax_ratio', 'pagerank', 'degree']]  
    
    logging.info("feature extraction done")
    print("feature extraction done")
    namespace.df=final_master


# #### Multiprocess Fn to score the network

# In[43]:


def network_score():
    manager=multiprocessing.Manager()
    feature_df1=pd.DataFrame()
    feature_ns1=manager.Namespace()
    feature_ns1.df=feature_df1
    feature_df2=pd.DataFrame()
    feature_ns2=manager.Namespace()
    feature_ns2.df=feature_df2
    feature_df3=pd.DataFrame()
    feature_ns3=manager.Namespace()
    feature_ns3.df=feature_df3
    feature_df4=pd.DataFrame()
    feature_ns4=manager.Namespace()
    feature_ns4.df=feature_df4
    feature_df5=pd.DataFrame()
    feature_ns5=manager.Namespace()
    feature_ns5.df=feature_df5
    feature_df6=pd.DataFrame()
    feature_ns6=manager.Namespace()
    feature_ns6.df=feature_df6
    final_master_split=np.array_split(final_master,6)
    f1 = multiprocessing.Process(target=feature_extraction, args=(final_master_split[0],feature_ns1))
    f2 = multiprocessing.Process(target=feature_extraction, args=(final_master_split[1],feature_ns2))
    f3 = multiprocessing.Process(target=feature_extraction, args=(final_master_split[2],feature_ns3))
    f4 = multiprocessing.Process(target=feature_extraction, args=(final_master_split[3],feature_ns4))
    f5 = multiprocessing.Process(target=feature_extraction, args=(final_master_split[4],feature_ns5))
    f6 = multiprocessing.Process(target=feature_extraction, args=(final_master_split[5],feature_ns6))

    f1.start()
    f2.start()
    f3.start()
    f4.start()
    f5.start()
    f6.start()
    f1.join()
    f2.join()
    f3.join()
    f4.join()
    f5.join()
    f6.join()

    final_master_final=pd.concat([feature_ns1.df,feature_ns2.df,feature_ns3.df,feature_ns4.df,feature_ns5.df,feature_ns6.df],sort=True)

    fn1=['reg_type_sc', 'cstn_bus_sc_g1', 'cstn_bus_sc_g2', 'cstn_bus_sc_oth', 'state_north', 'state_north_east', 
         'state_south', 'state_west', 'state_east', 'state_ut', 'ntr_pos_les', 'ntr_pos_sc_own', 'ntr_pos_sc_con', 
         'ntr_pos_sc_oth', 'is_migrated', 'age_of_gstin', 'gti_range', 'aadhar_linked', 'overall_score', 'isnil', 
         'r3b_sup_details_osup_nil_exmp_txval', 'r3b_sup_details_osup_nongst_txval', 'r3b_taxable_turnover', 
         'r3b_itc_availed', 'lgr_cash_utilized_3b', 'r1_elig_vs_r1_filed', 'r3b_elig_vs_r3b_filed', 
         'r3b_sup_details_osup_det_tax', 'r3b_sup_details_osup_zero_tax', 'r3b_4a_1_2_3_4', 
         'extra_txval', 'cash_vs_total', 'supply_loss', 'b2b_vs_total_tax_ratio', 'b2cl_vs_total_tax_ratio','pagerank',
         'degree']
    predicted_prob=best_model.predict_proba(final_master_final[fn1])
    proba_1=predicted_prob[:,1]
    logging.info("Prediction done")
    print("Prediction done",proba_1)
    final_master_final['prob']=proba_1
        ### Multiple avged path network features by Prob score.
    nw1=[ 'itc_passed_cnt', 'itc_utilized_cnt', 'missing_tax_cnt', 'high_itc_cnt', 'hi_b2c_cnt', 'suspicious_tran_cnt',
          'age_of_gstin_nw', 'suo_flag', 'cancel_flag', 'exporter_flag', 'cash_vs_total_nw', 'cnl_ewb_per', 
         'cnt_sensitive_hsn','state_cnt']

    final_master_final[nw1]=final_master_final[nw1].multiply(final_master_final['prob'], axis="index") 
    print("multiply by prob done ")
    logging.info("multiply by prob done")
    network_df=pd.DataFrame(final_master_final.groupby('root_gstin')[nw1].mean())   
    
    weights = np.array([7,7,6,6,6,7,6,7,4,6,8,2,5,7])    
    #score_arr = network_df*weights    
   
    network_df['network_score']=network_df.apply(lambda x: (x* weights).sum(), axis=1) 
    logging.info("scoring done")
    print("scoring done")    #
        #round(score_arr.sum(axis=1)[0],2) 
        #ntwk_score1['network_score']= round(score_arr.sum(axis=1)[0],2)    
    network_df['root_gstin']=network_df.index   
    return network_df,final_master_final


# #### Path Extraction 
# 

# In[44]:


def main(leads_gstin,master,no_path_ns,error_ns,error_paths_ns,h):
    try:
        master_tmp=pd.DataFrame()
        no_path_df=pd.DataFrame()
        error_df=pd.DataFrame()
        for g in leads_gstin:
            #print(g)
        #print("pid:{1},gstin:{2}".format(os.getpid(),g))
            df_paths=get_all_path_new(g,h,error_paths_ns)
            print("path extracted for",g)
            #current_time = datetime.datetime.now()
            #logging.info('execution start time:'+str(current_time))
            logging.info("path extracted for"+g)
            if df_paths.shape[0]==0:
                ntwk_score1={}
                ntwk_score1['gstin_id']=g
                ntwk_score1['network_score']= ["No Paths Found"] 
                no_path_df=no_path_df.append(ntwk_score1,ignore_index=True)
            else:
                df_paths=df_paths.set_index('index_new')
            #df_paths1=get_feat_proba_new(df_paths,df_path_feature,data_network_score)
                #print(df_paths) 
                master_tmp=master_tmp.append(df_paths, ignore_index=True)
   
        master.df=master_tmp
        no_path_ns.df=no_path_df
        
    except Exception as e:
        error={}
        error['gstin_id']=g
        error['error']= e
        error_df=error_df.append(error,ignore_index=True)
        error_ns.df=error_df


# #### Run of Main Function

# In[73]:


master_network_df=pd.DataFrame()
master_final_master_final=pd.DataFrame()

%%time
import multiprocessing
import subprocess
import os
import itertools
for i in range(0,count_of_df):
    start_index=i*1000
    last_index=start_index+1000
    if last_index>len(df_leads):
        last_index=len(df_leads)
    else:
        last_index=last_index
    gstin_path=df_leads[start_index:last_index]['gstin_id'].tolist()
    final_master, no_paths_final_df, error_final_df,error_paths_df = multiprocess_func(gstin_path)
    print("finished fetching paths")
    errors=pd.concat([error_final_df,error_paths_df], sort=True) 

    if (errors.shape[0]) > 0 :
        errors_gstin=errors.gstin_id.unique().tolist()
        #errors_gstin.remove('34NLEPS9553N1ZZ')
        #errors_gstin.remove('33AAFCI2906R1ZA')
        final_master_1, no_paths_final_df_1, error_final_df_1,error_paths_df_1 = multiprocess_func(errors_gstin)
        if final_master_1.shape[0]>0:
            final_master=final_master.append(final_master_1, ignore_index=True)
            no_paths_final_df=no_paths_final_df.append(no_paths_final_df_1, ignore_index=True)
    print("full extraction complete")
    final_master=final_master.fillna('NA')
    final_master['unique_gstin']=final_master.apply(lambda x: list(set(x['gstin_id'])), axis=1)  
    gstin_to_extract=pd.DataFrame(set(itertools.chain.from_iterable(final_master.unique_gstin)), columns=['gstin_id'])
    df_network=data_network_score.merge(gstin_to_extract, on='gstin_id', how='inner' )
    df_path_ft=df_path_feature.merge(gstin_to_extract, on='gstin_id', how='inner' )
    network_df,final_master_final=network_score()
    master_network_df=master_network_df.append(network_df)
    master_final_master_final=master_final_master_final.append(final_master_final)
    master_error_paths_df=master_error_paths_df.append(error_paths_df_1)

# In[ ]:


df_leads=['01AAAAI0050M1Z9']


# In[ ]:


gstin_list_not_to_be_taken=['27GIDPP2672A1ZC','34NLEPS9553N1ZZ','33AAFCI2906R1ZA','17ARTPP3068P1ZQ','29AAACW0315K1ZD',
                           '29DZDPK6396A1Z5','09ALGPS8092P1ZZ']


# In[ ]:


df_leads=df_leads.drop([x for x in gstin_list_not_to_be_taken if x in df_leads.gstin_id],axis=1)


# In[74]:


get_ipython().run_cell_magic('time', '', 'import multiprocessing\nimport subprocess\nimport os\nimport itertools\n\n    gstin_path=df_leads[start_index:last_index][\'gstin_id\'].tolist()\n    final_master, no_paths_final_df, error_final_df,error_paths_df = multiprocess_func(gstin_path)\n    print("finished fetching paths")\n    errors=pd.concat([error_final_df,error_paths_df], sort=True) \n\n    if (errors.shape[0]) > 0 :\n        errors_gstin=errors.gstin_id.unique().tolist()\n        #errors_gstin.remove(\'34NLEPS9553N1ZZ\')\n        #errors_gstin.remove(\'33AAFCI2906R1ZA\')\n        final_master_1, no_paths_final_df_1, error_final_df_1,error_paths_df_1 = multiprocess_func(errors_gstin)\n        if final_master_1.shape[0]>0:\n            final_master=final_master.append(final_master_1, ignore_index=True)\n            no_paths_final_df=no_paths_final_df.append(no_paths_final_df_1, ignore_index=True)\n    print("full extraction complete")\n    final_master=final_master.fillna(\'NA\')\n    final_master[\'unique_gstin\']=final_master.apply(lambda x: list(set(x[\'gstin_id\'])), axis=1)  \n    gstin_to_extract=pd.DataFrame(set(itertools.chain.from_iterable(final_master.unique_gstin)), columns=[\'gstin_id\'])\n    df_network=data_network_score.merge(gstin_to_extract, on=\'gstin_id\', how=\'inner\' )\n    df_path_ft=df_path_feature.merge(gstin_to_extract, on=\'gstin_id\', how=\'inner\' )\n    network_df,final_master_final=network_score()\n    #master_network_df=master_network_df.append(network_df)\n    #master_final_master_final=master_final_master_final.append(final_master_final)\n    #master_error_paths_df=master_error_paths_df.append(error_paths_df_1)')


# In[42]:


#calculation of score bucket
network_df_total=network_df.copy()
network_df_total['score_bucket']=network_df_total.apply(lambda x : "Risky" if x.network_score<=5 else ("High Risky" if x.network_score>5 and x.network_score<=10 else "Very High Risky"), axis=1)


# In[43]:


#rounding the network score for diplay in Dashboard and UI
network_df_total.network_score=network_df_total.network_score.round(2)
network_df_total.head()


# In[77]:


#network_df_total.score_bucket.value_counts()


# In[46]:


#network_df_total.to_csv("network_df_total.csv")


# In[49]:


#saving the file for analysis of probability score
#master_final_master_final_prt=master_final_master_final[['root_gstin','unique_gstin','prob','suo_flag']]
#master_final_master_final_prt.to_csv("final_master_final_total_part.csv")


# ### Network score transfer to hive table

# In[95]:


#network_score_previous=pd.read_sql('select gstin_id as gstin_id,network_score as network_score,score_bucket as score_bucket from genome_distribution.network_score_hstr',conn)


# In[96]:


#network_score_previous.shape


# In[97]:


#network_score_previous.head()


# In[98]:


network_pq=pd.DataFrame(network_df_total[['network_score','score_bucket']])
#network_pq=network_total_score_new[['root_gstin','network_score']]
network_pq.reset_index(inplace=True)
network_pq.columns=['gstin_id','network_score','score_bucket']


# In[107]:


#total_network_score=pd.concat([network_pq,network_score_previous],axis=0)
#network_pq=total_network_score.drop_duplicates(keep='first')


# In[112]:


#network_pq.to_parquet('network_score_final', engine='pyarrow',compression='snappy',index=None) 


# In[113]:


#network_pq.to_parquet('network_score_final', engine='pyarrow',compression='snappy',index=None) 
#res=subprocess.run(["curl", "-L","--negotiate", "-u:",  "-X",  "PUT", "-T", "/home/such4579/Network_score_incremental_code/network_score_final","http://10.0.8.51:9870/webhdfs/v1/user/hive/warehouse/genome_distribution.db/network_score/network_score?user.name=genome_user&op=CREATE&overwrite=true"])
#res


# In[ ]:




