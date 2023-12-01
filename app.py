from flask import Flask, render_template, request
import allmodels as all
import modelcreationfunctions as m
import databasecreationfunctions as d
import pandas as pd

def sortkey(elem):
    return elem[-1]

app = Flask(__name__)

@app.route("/")
def home():
    df = d.create_df()
    df2 = d.random_df(1000)
    d.save(df2,"C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\templates\\records.html","C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\records.csv")
    d.save(df,"C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\templates\\patients_records.html","C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\patients_records.csv")
    return render_template("index.html")

@app.route("/nurse")
def nurse():
    return render_template("Nurse.html")

@app.route("/head")
def head():
    return render_template("Head.html")

@app.route("/medignosisT")
def medT():
    return render_template("MedignosisT.html")

@app.route("/Headact")
def headact():
    return render_template("Head_action.html")

@app.route("/Headact",methods=["POST"])
def getvalue4():
    df = d.load_df("records.csv")
    action = request.form["act"]
    if action == "UCI":
        n = d.df_uci(df)
        d.save(n,"C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\templates\\UCI_records.html","UCI_records.csv")
        return render_template("UCI_records.html")
    if action == "Admitidos":
        n = d.df_internos(df)
        d.save(n,"C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\templates\\int_records.html","int_records.csv")
        return render_template("int_records.html")
    if action == "Antiguos":
        n = d.df_pacientesantiguos(df)
        d.save(n,"C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\templates\\ant_records.html","ant_records.csv")
        return render_template("ant_records.html")
    else:
        n = d.df_internos(df)
        d.save(n,"C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\templates\\in_records.html","in_records.csv")
        return render_template("in_records.html")

@app.route("/Nurseact")
def nurseact():
    return render_template("Nurse_action.html")

@app.route("/Nurseact", methods=["POST"])
def getvalue3():
    df = d.load_df("patients_records.csv")
    ID = request.form["DI"]
    action = request.form["act"]

    if action == "Eliminar paciente":
        m = d.remove_patient(df,ID)
        d.save(df,"C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\templates\\patients_records.html","patients_records.csv")
        return render_template("message.html", phrase=m)
    elif action == "Buscar paciente":
        n = d.get_patient(df,ID)
        if type(n)!= str:
            m = "UCI: ",n["uci"]," ADMITIDO: ",n["admitido"]," DADO EN ALTA: ",n["alta"]," SEXO: ",n["Sex"]," EDAD: ",n["Age"]," NÚMERO LLEGADA: ",n["Patients number per hour"]," TRANSPORTE: ",n["Arrival mode"]," HERIDO: ",n["Injury"]," QUEJA PRINCIPAL: ",n["Chief_complain"]," ESTADO MENTAL: ",n["Mental"]," DOLOR: ",n["Pain"]," ESCALA DOLOR (1-10): ",n["NRS_pain"]," PRESIÓN SANGUÍNEA SISTÓLICA: ",n["SBP"]," PRESIÓN SANGUÍNEA DIASTÓLICA: ",n["DBP"]," FRECUENCIA CARDÍACA: ",n["HR"]," FRECUENCIA RESPIRATORIA: ",n["RR"]," TEMPERATURA: ",n["BT"]," SAT. OXÍGENO: ",n["Saturation"]," TRIAGE: ",n["KTAS_RN"]," DIAGNÓSTICO: ",n["Diagnosis in ED"]," DESICIÓN MÉDICA: ",n["Disposition"]," DURACIÓN ESTIMADA: ",n["Length of stay_min"]," DURACIÓN CHEQUEO: ",n["KTAS duration_min"]," PREDICCIÓN TRIAGE: ",n["ktas_pred"]," PREDICCIÓN MISTRIAGE: ",n["mistriage_pred"]
            return render_template("message.html", phrase=m)
        else: 
            return render_template("message.html",phrase=n)
    elif action == "Dar de alta a paciente":
        m = d.dar_de_alta(df,ID)
        d.save(df,"C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\templates\\patients_records.html","patients_records.csv")
        return render_template("message.html",phrase=m)
    elif action == "Entrar paciente a UCI":
        m = d.entrada_uci(df,ID)
        d.save(df,"C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\templates\\patients_records.html","patients_records.csv")
        return render_template("message.html",phrase=m)
    elif action == "Sacar paciente de UCI":
        m = d.salida_uci(df,ID)
        d.save(df,"C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\templates\\patients_records.html","patients_records.csv")
        return render_template("message.html",phrase=m)
    else: 
        m = d.admitir(df,ID)
        d.save(df,"C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\templates\\patients_records.html","patients_records.csv")
        return render_template("message.html",phrase=m)

@app.route("/medignosisT", methods=["POST"])
def getvalue1():
    df = d.load_df("patients_records.csv")
    nombre = request.form["Nombre"]
    ID = request.form["DI"]
    Edad = request.form["Edad"]
    Numero = request.form["Numero"]
    Sexo = request.form["Sexo"]
    NumeroL = request.form["Numero llegada"]
    MC = request.form["MC"]
    transport = request.form["transport"]
    mental = request.form["mental"]
    hurt = request.form["hurt"]
    hurtf = request.form["hurt-f"]
    hurts = int(request.form["hurt-s"])
    HR = request.form["HR"]
    BT = request.form["BT"]
    RR = request.form["RR"]
    SBP = request.form["SBP"]
    DBP = request.form["DBP"]
    OS = request.form["OS"]
    grav = request.form["grav"]
    DE = request.form["DE"]
    DR = request.form["DR"]
    decision = request.form["decision"]
    D = request.form["D"]

    if Sexo == "Mujer": 
        Sexo = 2
    else: 
        Sexo = 1

    if transport == "Caminando":
        transport = 1
    elif transport == "Ambulancia pública":
        transport = 2
    elif transport == "Carro particular":
        transport = 3
    elif transport == "Ambulancia privada":
        transport = 4
    elif transport == "Transporte público (policia)":
        transport = 5
    elif transport == "Silla de ruedas":
        transport = 6
    else: 
        transport = 7
    
    if mental == "Alerta":
        mental = 1
    elif mental == "Responde a estímulos verbales":
        mental = 2
    elif mental == "Responde a estímulos de dolor":
        mental = 3
    else: 
        mental = 4
    
    if hurt == "Si": 
        hurt = 2
    else: 
        hurt = 1
    
    if hurtf == "Si":
        hurtf = 1
    else: 
        hurtf = 0
    
    if grav == "Signos vitales":
        grav = 1
    elif grav == "Exámen físico":
        grav = 2
    elif grav == "Paciente psiquiátrico":
        grav = 3
    elif grav == "Dolor":
        grav = 4
    elif grav == "Problemas mentales":
        grav = 5
    elif grav == "Enfermedad subyacente":
        grav = 6
    else: 
        grav = 7

    if decision == "Dar de alta":
        decision = 1
    elif decision == "Admisión en sala":
        decision = 2
    elif decision == "UCI":
        decision = 3
    elif decision == "Dado de alta CRM":
        decision = 4
    elif decision == "Transferido":
        decision = 5
    else: 
        decision = 6

    no_ID = [1,float(Sexo),float(Edad),float(NumeroL),float(transport),float(hurt),MC,float(mental),float(hurtf),float(hurts),float(SBP),float(DBP),float(HR),float(RR),float(BT),float(OS),1,D,float(decision),1,float(grav),float(DE)*60,float(DR),0]
    data = m.prepare(no_ID,False,True,"data.csv")
    prediction = all.ktas().predict(data)
    no_ID2 = [1,float(Sexo),float(Edad),float(NumeroL),float(transport),float(hurt),MC,float(mental),float(hurtf),float(hurts),float(SBP),float(DBP),float(HR),float(RR),float(BT),float(OS),float(prediction[0]),D,float(decision),1,float(grav),float(DE)*60,float(DR),0]
    data2 = m.prepare(no_ID2,True,False,"data.csv")
    prediction2 = all.mistriage().predict(data2)
    df = d.add_patient(df,ID,0,0,0,Sexo,Edad,NumeroL,transport,hurt,MC,mental,hurtf,hurts,SBP,DBP,HR,RR,BT,OS,prediction[0],D,decision,0,grav,DE,DR,0,prediction[0],prediction2[0])
    d.save(df,"C:\\Users\\USUARIO\\OneDrive - Universidad de los andes\\Documentos\\Uniandes\\7mo\\Reto empresarial\\Medignosis_website\\manurivasg.github.io\\templates\\patients_records.html","patients_records.csv")
    return render_template("patients_records.html")

@app.route("/medignosisT2", methods=["POST"])
def getvalue2():
    nombre = request.form["Nombre"]
    ID = request.form["DI"]
    Edad = request.form["Edad"]
    Numero = request.form["Numero"]
    Sexo = request.form["Sexo"]
    NumeroL = request.form["Número llegada"]
    MC = request.form["MC"]
    transport = request.form["transport"]
    mental = request.form["mental"]
    hurt = request.form["hurt"]
    hurtf = request.form["hurt-f"]
    hurts = int(request.form["hurt-s"])
    HR = request.form["HR"]
    BT = request.form["BT"]
    RR = request.form["RR"]
    SBP = request.form["SBP"]
    DBP = request.form["DBP"]
    OS = request.form["OS"]
    grav = request.form["grav"]
    DE = request.form["DE"]
    DR = request.form["DR"]
    triage = int(request.form["triage"])
    decision = request.form["decision"]
    D = request.form["D"]

    if Sexo == "Mujer": 
        Sexo = 2
    else: 
        Sexo = 1

    if transport == "Caminando":
        transport = 1
    elif transport == "Ambulancia pública":
        transport = 2
    elif transport == "Carro particular":
        transport = 3
    elif transport == "Ambulancia privada":
        transport = 4
    elif transport == "Transporte público (policia)":
        transport = 5
    elif transport == "Silla de ruedas":
        transport = 6
    else: 
        transport = 7
    
    if mental == "Alerta":
        mental = 1
    elif mental == "Responde a estímulos verbales":
        mental = 2
    elif mental == "Responde a estímulos de dolor":
        mental = 3
    else: 
        mental = 4
    
    if hurt == "Si": 
        hurt = 2
    else: 
        hurt = 1
    
    if hurtf == "Si":
        hurtf = 1
    else: 
        hurtf = 0
    
    if grav == "Signos vitales":
        grav = 1
    elif grav == "Exámen físico":
        grav = 2
    elif grav == "Paciente psiquiátrico":
        grav = 3
    elif grav == "Dolor":
        grav = 4
    elif grav == "Problemas mentales":
        grav = 5
    elif grav == "Enfermedad subyacente":
        grav = 6
    else: 
        grav = 7

    if decision == "Dar de alta":
        decision = 1
    elif decision == "Admisión en sala":
        decision = 2
    elif decision == "UCI":
        decision = 3
    elif decision == "Dado de alta CRM":
        decision = 4
    elif decision == "Transferido":
        decision = 5
    else: 
        decision = 6
    
    no_ID = [1,float(Sexo),float(Edad),float(NumeroL),float(transport),float(hurt),MC,float(mental),float(hurtf),float(hurts),float(SBP),float(DBP),float(HR),float(RR),float(BT),float(OS),float(triage),D,float(decision),1,float(grav),float(DE)*60,float(DR),0]
    data = m.prepare(no_ID,True,False,"data.csv")
    prediction = all.mistriage().predict(data)
    phrase="Predicción de mistriage: "+prediction[0]
    return render_template("message.html",phrase=phrase)


@app.route("/medignosisT2")
def medT2():
    return render_template("MedignosisT2.html")

@app.route("/patients")
def patients():
    return render_template("patients.html")

@app.route("/patients",methods=["POST"])
def getvalue5():
    ID = request.form["DI"]
    Edad = request.form["Edad"]
    Sexo = request.form["Sexo"]
    NumeroL = 0
    MC = request.form["MC"]
    transport = request.form["transport"]
    mental = request.form["mental"]
    hurt = request.form["hurt"]
    hurtf = request.form["hurt-f"]
    hurts = int(request.form["hurt-s"])
    HR = request.form["HR"]
    BT = request.form["BT"]
    RR = 16
    SBP = 110
    DBP = 70
    OS = 98
    grav = 1
    DE = 4
    DR = 15
    triage = 3
    decision = 1
    D = "NAN"

    if Sexo == "Mujer": 
        Sexo = 2
    else: 
        Sexo = 1

    if transport == "Caminando":
        transport = 1
    elif transport == "Ambulancia pública":
        transport = 2
    elif transport == "Carro particular":
        transport = 3
    elif transport == "Ambulancia privada":
        transport = 4
    elif transport == "Transporte público (policia)":
        transport = 5
    elif transport == "Silla de ruedas":
        transport = 6
    else: 
        transport = 7
    
    if mental == "Alerta":
        mental = 1
    elif mental == "Responde a estímulos verbales":
        mental = 2
    elif mental == "Responde a estímulos de dolor":
        mental = 3
    else: 
        mental = 4
    
    if hurt == "Si": 
        hurt = 2
    else: 
        hurt = 1
    
    if hurtf == "Si":
        hurtf = 1
    else: 
        hurtf = 0
    
    if grav == "Signos vitales":
        grav = 1
    elif grav == "Exámen físico":
        grav = 2
    elif grav == "Paciente psiquiátrico":
        grav = 3
    elif grav == "Dolor":
        grav = 4
    elif grav == "Problemas mentales":
        grav = 5
    elif grav == "Enfermedad subyacente":
        grav = 6
    else: 
        grav = 7

    if decision == "Dar de alta":
        decision = 1
    elif decision == "Admisión en sala":
        decision = 2
    elif decision == "UCI":
        decision = 3
    elif decision == "Dado de alta CRM":
        decision = 4
    elif decision == "Transferido":
        decision = 5
    else: 
        decision = 6
    
    no_ID = [1,float(Sexo),float(Edad),float(NumeroL),float(transport),float(hurt),MC,float(mental),float(hurtf),float(hurts),float(SBP),float(DBP),float(HR),float(RR),float(BT),float(OS),float(triage),D,float(decision),1,float(grav),float(DE)*60,float(DR),0]
    data = m.prepare(no_ID,False,True,"data.csv")
    prediction = all.ktas().predict(data)
    phrase="Predicción de triage preliminar: ",prediction[0]
    return render_template("message.html",phrase=phrase)

@app.route("/Headgraph")
def graph():
    return render_template("Head_graph.html")

@app.route("/Headgraph",methods=["POST"])
def getvalues6():
    df = d.load_df("records.csv")
    action = request.form["act"]
    if action == "UCI por edades":
        bins = [0, 20, 40, 60, 80, 100, df['Age'].max() + 10]
        df['Age_Category'] = pd.cut(df['Age'], bins=bins)
        df = df.dropna(subset=['Age_Category'])
        df['Age_Category'] = pd.IntervalIndex(df['Age_Category']).sort_values()
        category_counts = df.groupby('Age_Category')['uci'].sum().tolist()
        age_categories = df['Age_Category'].unique().astype(str).tolist() 
        return render_template('bar_plot.html', age_categories=age_categories, category_counts=category_counts)
    else:
        bins = [0, 20, 40, 60, 80, 100, df['Age'].max() + 10]
        df['Age_Category'] = pd.cut(df['Age'], bins=bins)
        df = df.dropna(subset=['Age_Category'])
        df['Age_Category'] = pd.IntervalIndex(df['Age_Category']).sort_values()
        category_counts = df.groupby('Age_Category')['admitido'].sum().tolist()
        age_categories = df['Age_Category'].unique().astype(str).tolist() 
        return render_template('bar_plot copy.html', age_categories=age_categories, category_counts=category_counts)


if __name__ == "__main__":
    app.run(debug=True)