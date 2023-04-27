# WoT Project: SarcopeniaApp - Back-End in Cloud - Norberti Simone

L’obiettivo del progetto è la realizzazione di un sistema IoT low-cost e low-power che si occupa della raccolta, elaborazione ed analisi dei dati diagnostici di pazienti affetti da una malattia neurodegenerativa, la Sarcopenia, con la successiva visualizzazione delle informazioni estrapolate. Il punto di forza di questo progetto è che tale processo di monitoraggio e diagnostica viene effettuato direttamente da casa, a domicilio, senza che il paziente debba recarsi fisicamente presso una struttura ospedaliera. Inoltre, il medico ha la possibilità di effettuare monitoraggio e diagnostica direttamente nel suo studio da remoto grazie all’ausilio e al supporto del sistema stesso. La creazione di questo sistema è reso possibile grazie all’utilizzo di tecnologie IoT e di tecniche di Intelligenza Artificiale che, automatizzando il processo di monitoraggio e diagnostica, rendendo il tracciamento della malattia più semplice, accurato, senza mancate misurazioni, a basso costo e a basso consumo energetico.

Simulatore sensori: https://github.com/UniSalento-IDALab-IoTCourse-2021-2022/wot-project-raspberry-SimoneNorberti

App Android paziente: https://github.com/UniSalento-IDALab-IoTCourse-2021-2022/wot-project-androidapp-SimoneNorberti

Back-End (Cloud): https://github.com/UniSalento-IDALab-IoTCourse-2021-2022/wot-project-BackEnd-SimoneNorberti

Front-End (Cloud): https://github.com/UniSalento-IDALab-IoTCourse-2021-2022/wot-project-FrontEnd-SimoneNorberti


## Back-End in Cloud
In questo progetto viene utilizzato Oracle Cloud, in particolare una Computing Instance con shape VM.Standard.E2.1.Micro.
Una volta creata la VM, che nel mio caso ha indirizzo 152.70.169.171, modifichiamo la VNIC aggiungengo nella Security List l'accesso alla porta TCP/5000 da qualunque indirizzo esterno (0.0.0.0/0), porta sulla quale verrà esposto il server Back-End.

Durante la fase di creazione dell'istanza, vengono salvate due chiavi sul PC, una privata e una pubblica. Utilizziamo la chiave privata per accedere tramite SSH al server (il default user è ```opc```):
```
chmod 400 ./ssh-key-2023-04-24.key
ssh -i ssh-key-2023-04-24.key opc@152.70.169.171
```

Utilizziamo git per scaricare il codice Python sul server:
```
git clone https://github.com/UniSalento-IDALab-IoTCourse-2021-2022/wot-project-BackEnd-SimoneNorberti
```

Tramite il comando firewall-cmd apriamo la porta TCP 5000
```
sudo firewall-cmd --zone=public --add-port=5000/tcp
sudo firewall-cmd --runtime-to-permanent
sudo firewall-cmd –-reload
```

Installiamo l'interprete Python (versione 3.7 o superiore):
```
sudo yum install python3
sudo pip3 install pandas, flask, flask_cors, pymongo, scikit-learn
pyhton3.7 app.py
```

### Database
In questo progetto viene utilizzato un DBaaS (DataBase as a Service), in particolare Atlas MongoDB, un Cloud di MongoDB utilizzabile gratuitamente.
Una volta su https://cloud.mongodb.com/ ed aver effettuato l'accesso, creare un nuovo progetto, un nuovo Daatabase e abilitare l'accesso dall'IP del server Oracle (es. 152.70.169.171)
