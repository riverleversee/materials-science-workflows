import numpy as np 
filenamesend=['11','12','13','14','15','16','17','18','19','20','21','22','23','23','25','26','27','28','29']

filenamesbegin=['../../../qe-7.0/PW/MeshTestingBenzene/HydroFilesUnfix/results/benzenemesh777press']

filecoordlist=['']*len(filenamesend)

for i in range(len(filenamesend)):
	filecoordlist[i]=filenamesbegin[0]+filenamesend[i]+'gpa.out'

filestartlist=['relaxbase.sh']




pressurelist=['110.0,', '120.0,','130.0,','140.0,','150.0,', '160.0,','170.0,','180.0,','190.0,','200.0,','210.0,','220.0,','230.0,','240.0,','250.0,','260.0,','270.0,','280.0,','290.0,']
fileendlist=['hydro11.sh','hydro12.sh','hydro13.sh','hydro14.sh','hydro15.sh','hydro16.sh','hydro17.sh','hydro18.sh','hydro19.sh','hydro20.sh','hydro21.sh','hydro22.sh','hydro23.sh','hydro24.sh','hydro25.sh','hydro26.sh','hydro27.sh','hydro28.sh','hydro29.sh']



for z in range(len(fileendlist)):

    fileorigcalc=filestartlist[0]
    fileoriginitbase=filecoordlist[z]

    
    filenew=fileendlist[z]

    fileoriginit=fileoriginitbase
    print("calc="+fileorigcalc)	
    print("newname="+filenew)
    print("startpos="+fileoriginit)
    fnew=open(filenew,"w")
    foldinit=open(fileoriginit,"r")
    foldcalc=open(fileorigcalc,"r")
    filelinespos=foldinit.readlines()
    filelinescalc=foldcalc.readlines()
    
    
#    presstrigger='  press'
#    for i in range(len(filelinescalc)):
#        if filelinescalc[i][0:len(presstrigger)]==presstrigger:
#            changepressline=i
#            break

#    newpressline=''
#    k=0
#    for i in range(len(filelinescalc[changepressline])):
#        if filelinescalc[changepressline][i].isnumeric() and k<3:
#            newpressline +=newpressure[k]
#            k=k+1
#        else:
#            newpressline +=filelinescalc[changepressline][i]
#    filelinescalc[changepressline]=newpressline    
    
    calccelldmtrigger='&SYSTEM'
    for i in range(len(filelinescalc)):
        if filelinescalc[i][0:len(calccelldmtrigger)]==calccelldmtrigger:
            changecalccelldmline=i+5
            break
    
    
    initcelldmtrigger='Begin final coordinates'
    for i in range(len(filelinespos)):
        if filelinespos[len(filelinespos)-i-1][0:len(initcelldmtrigger)]==initcelldmtrigger :
            pullinitcelldmline=len(filelinespos)-i-1-26
            pullinitcellatomposline=len(filelinespos)-i-1+9+1
            break
        else:
       	 pullinitcelldmline=0;
    if pullinitcelldmline==0:
    	print("cannot find final coordinates")        
        
       
    calccellatompostrigger= 'ATOMIC_POSITIONS crystal'
    for i in range(len(filelinescalc)):
        if filelinescalc[i][0:len(calccellatompostrigger)]==calccellatompostrigger:
            changecalcatomposline=i+1
            break        
    for i in range(4):   
        filelinescalc[changecalccelldmline+i]=filelinespos[pullinitcelldmline+i]
    
    for i in range(24):

    	 filelinescalc[changecalcatomposline+i]=filelinespos[pullinitcellatomposline+i]


    pressuretrigger='  press='
    for i in range(len(filelinescalc)):
        if filelinescalc[i][0:len(pressuretrigger)]==pressuretrigger:
            changepressline=i
            break       
    newpressline='  press='+pressurelist[z]+'\n'
    filelinescalc[changepressline]=newpressline

    
    filename1trigger='cat >'
    filename2trigger='  prefix ='    
    filename3trigger='$PW_COMMAND <'

    for i in range(len(filelinescalc)):
        if filelinescalc[i][0:len(filename1trigger)]==filename1trigger:
            changefileline1=i
        if filelinescalc[i][0:len(filename2trigger)]==filename2trigger:
            changefileline2=i
        if filelinescalc[i][0:len(filename3trigger)]==filename3trigger:
            changefileline3=i    
            break
	
    newfilename1line='cat >   '+filenew[0:-3]+'.in  << EOF \n'
 
    newfilename2line="  prefix = '"+filenew[0:-3]+"'\n"
    newfilename3line="$PW_COMMAND < "+filenew[0:-3]+".in > "+filenew[0:-3]+".out\n"


    filelinescalc[changefileline1]=newfilename1line  
    filelinescalc[changefileline2]=newfilename2line  
    filelinescalc[changefileline3]=newfilename3line  
    fnew.writelines(filelinescalc)
    print('done')


