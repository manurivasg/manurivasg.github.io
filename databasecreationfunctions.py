import pickle as pkl
import pandas as pd
import numpy as np
import time as time
import pytz
from datetime import datetime
import chardet

#Create, Load, Add new Entry, Save

def create_df():
    return pd.DataFrame()

def load_df(path):
    df=pd.read_csv(path,sep=";",index_col=0)
    df.index = df.index.astype(str)
    return df

def save(df,pathhtml,pathcsv):
    df.to_html(pathhtml)
    df.to_csv(pathcsv,sep=";")

def remove_patient(df,ID):
    if not df.empty:
        if ID not in df.index.values:
            return "El paciente a remover no se encuentra en la base de datos."
        else:
            df.drop(index=ID,inplace=True)
            return "El paciente fue removido de la base de datos exitosamente."
    else:
        return "La base de datos se encuentra vacía. Por ende, el paciente a remover no se encuentra en la base de datos."

def get_patient(df,ID):
    if not df.empty:
        if ID not in df.index.values:
            return "El paciente a buscar no se encuentra en la base de datos."
        else:
            return df.loc[ID]
    else:
        return "La base de datos se encuentra vacía. Por ende, el paciente a buscar no se encuentra en la base de datos."

def add_patient(df:pd.DataFrame,ID,uci,admitido,alta,gender,age, num_patient, arrival_v, hurt, qp, metal, pain, pain_sc, sbp, dbp, hr, rr, bt, sat, triage, diag, disp, ktas_expert, err_gr, dur, ktas_dur,mistriage,ktas_pred,mistriage_pred):
    if not df.empty:
        indexes=list(df.index.values)
    else:
        indexes=[]
    index=[ID]
    if ID in indexes:
        numvisit=df.index.value_counts().astype(int).loc[ID]+1
    else:
        numvisit=1

    current_time_utc = time.time()
    utc_datetime = datetime.utcfromtimestamp(current_time_utc)
    colombia_timezone = pytz.timezone('America/Bogota')
    fechallegada = pytz.utc.localize(utc_datetime).astimezone(colombia_timezone)

    fechasalida=False
    cols=['Group', 'Sex', 'Age', 'Patients number per hour', 'Arrival mode',
       'Injury', 'Chief_complain', 'Mental', 'Pain', 'NRS_pain', 'SBP', 'DBP',
       'HR', 'RR', 'BT', 'Saturation', 'KTAS_RN', 'Diagnosis in ED',
       'Disposition', 'KTAS_expert', 'Error_group', 'Length of stay_min',
       'KTAS duration_min', 'mistriage']
    
    data={"ktas_pred":ktas_pred,"mistriage_pred":mistriage_pred,"fechallegada":[fechallegada],"fechasalida":[fechasalida],"uci":[uci],"admitido":[admitido],"alta":[alta],"numvisit":[numvisit],cols[0]:[1], cols[1]: [gender], cols[2]: [age], cols[3]:[num_patient], cols[4]: [arrival_v], cols[5]: [hurt], cols[6]: [qp], cols[7]: [metal], cols[8]: [pain], cols[9]:[pain_sc], cols[10]: [sbp], cols[11]: [dbp], cols[12]: [hr], cols[13]:[rr], cols[14]: [bt], cols[15]: [sat], cols[16]:[triage], cols[17]: [diag], cols[18]: [disp], cols[19]:[ktas_expert], cols[20]: [err_gr], cols[21]: [dur], cols[22]:[ktas_dur],cols[23]:[mistriage]}
    df2 = pd.DataFrame(data,index=pd.Index([ID]))
    df3 = pd.concat([df,df2])
    df3.sort_values(inplace=True,by="ktas_pred",axis=0)
    return df3

#Funciones Enfermera

def dar_de_alta(df,ID):
    if not df.empty:
        if ID in df.index.values:
            df.loc[ID,"alta"]=1
            df.loc[ID,"admitido"]=0
            df.loc[ID,"uci"]=0
            current_time_utc = time.time()
            utc_datetime = datetime.utcfromtimestamp(current_time_utc)
            colombia_timezone = pytz.timezone('America/Bogota')
            fechasalida = pytz.utc.localize(utc_datetime).astimezone(colombia_timezone)
            df.loc[ID,"fechasalida"]=fechasalida
            return "El paciente fue dado de alta exitosamente."
        else:
            return "El paciente no se encuentra en la base de datos."
    else:
        return "La base de datos se encuentra vacía."

def entrada_uci(df,ID):
    if not df.empty:
        if ID in df.index.values:
            df.loc[ID,"uci"]=1
            return "El paciente fue ingresado a la UCI exitosamente."
        else:
            return "El paciente no se encuentra en la base de datos."
    else:
        return "La base de datos se encuentra vacía."

def salida_uci(df,ID):
    if not df.empty:
        if ID in df.index.values:
            df.loc[ID,"uci"]=0
            return "El paciente fue removido de la UCI exitosamente."
        else:
            return "El paciente no se encuentra en la base de datos."
    else:
        return "La base de datos se encuentra vacía."

def admitir(df,ID):
    if not df.empty:
        if ID in df.index.values:
            df.loc[ID,"admitido"]=1
            return "El paciente fue removido de la UCI exitosamente."
        else:
            return "El paciente no se encuentra en la base de datos."
    else:
        return "La base de datos se encuentra vacía."


#Separación de datos

def df_uci(df)->pd.DataFrame:
    return df.loc[df["uci"]==1]

def df_internos(df)->pd.DataFrame:
    return df.loc[df["admitido"]==1]

def df_pacientesantiguos(df)->pd.DataFrame:
    return df.loc[df["alta"]==1]

def df_hospital(df)->pd.DataFrame:
    return df.loc[df["alta"]==0]

#Generate Random Data

def random_df(n):
    df=create_df()
    mistriagelist= ["Under Triage","Normal Triage", "Over Triage"]
    for i in range(0,n):
        ID = "1" + str(np.random.randint(100000000,999999999))
        uci=np.random.randint(0,2)
        admitido=np.random.randint(0,2)
        alta=np.random.randint(0,2)
        gender=np.random.randint(1,3)
        age=np.random.randint(0,111)
        num_patient=np.random.randint(1,57)
        arrival_v=np.random.randint(1,8)
        hurt=np.random.randint(1,2)
        qp="Consulta"
        metal=np.random.randint(1,5)
        pain=np.random.randint(1,3)
        pain_sc=np.random.randint(1,11)
        sbp=np.random.randint(70,200)
        dbp=np.random.randint(40,140)
        hr=np.random.randint(40,200)
        rr=np.random.randint(4,40)
        bt=(1/10)*np.random.randint(345,410)
        sat=(1/10)*np.random.randint(908,988)
        diag="Diagnóstico"
        disp=np.random.randint(1,7)
        err_gr=np.random.randint(1,8)
        dur=np.random.randint(4,500)
        ktas_dur=np.random.randint(5,25)
        triage=np.random.randint(1,6)
        ktas_expert=np.random.randint(1,6)
        mistriageindex=np.random.randint(0,3)
        mistriage=mistriagelist[mistriageindex]
        ktas_pred=np.random.randint(1,6)
        mistriageindex=np.random.randint(0,3)
        mistriage_pred=mistriagelist[mistriageindex]
        df=add_patient(df,ID,uci,admitido,alta,gender,age,num_patient,arrival_v,hurt,qp,metal,pain,pain_sc,sbp,dbp,hr,rr,bt,sat,triage,diag,disp,ktas_expert,err_gr,dur,ktas_dur,mistriage,ktas_pred,mistriage_pred)
    return df