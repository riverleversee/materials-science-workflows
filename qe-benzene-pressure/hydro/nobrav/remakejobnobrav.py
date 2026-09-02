"""Build ibrav=0 (nobrav) hydrostatic QE scripts from brav hydro .out results.

Takes CELL_PARAMETERS / positions from brav hydro{P}.out and writes
hydronobrav{P}.sh with matching press= for free-cell hydrostatic checks.
"""
import numpy as np 
filestartlist=['relaxbase.sh']

filecoordlist=['../brav/results/hydro10.out','../brav/results/hydro15.out','../brav/results/hydro20.out','../brav/results/hydro25.out','../brav/results/hydro30.out']
filestartlist=['relaxbase.sh']*len(filecoordlist)
        
pressurelist=['100.0,','150.0,','200.0,','250.0,','300.0,']

fileendlist=['hydronobrav10.sh','hydronobrav15.sh','hydronobrav20.sh','hydronobrav25.sh','hydronobrav30.sh']

for z in range(len(fileendlist)):

    fileorigcalc=filestartlist[z]
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
    



 #finds lattice alat and atompos    
    initcelldmtrigger='Begin final coordinates'
    pullinitcelldmline=-10
    pullinitcellatomposline=-10
    for i in range(len(filelinespos)):
        if filelinespos[len(filelinespos)-i-1][0:len(initcelldmtrigger)]==initcelldmtrigger :
            pullinitcelldmline=len(filelinespos)-i-1+4
            pullinitcellatomposline=len(filelinespos)-i-1+10
            break
    firstnumeric=1
    lastnumeric=1
    linecurrent=filelinespos[pullinitcelldmline]
    for i in range(len(filelinespos[pullinitcelldmline])): 
        if linecurrent[i].isnumeric() and firstnumeric==1:
            firstnumeric=0
            alatstart=i-1
        if linecurrent[len(linecurrent)-i-1].isnumeric() and lastnumeric==1:
            lastnumeric=0
            alatend=len(linecurrent)-i-1+1

    alat=float(linecurrent[alatstart:alatend])
#finds where to put in the alat    
    calccelldmtrigger='  celldm(1) ='
    changecalccelldmline=-10
    for i in range(len(filelinescalc)):
        if filelinescalc[i][0:len(calccelldmtrigger)]==calccelldmtrigger:
            changecalccelldmline=i
            print('dm1 set')
            break
    #Changes alat
    filelinescalc[changecalccelldmline]="  celldm(1) ="+f"{alat}, \n"
    
    calccellparamline="CELL_PARAMETERS alat"
    changecalccellparamline=-10
    for i in range(len(filelinescalc)):
        if filelinescalc[i][0:len(calccellparamline)]==calccellparamline:
            changecalccellparamline=i+1
            break    

    #linecurrent=filelinespos[pullinitcelldmline+1+j]
    filelinescalc[changecalccellparamline]=filelinespos[pullinitcelldmline+1]
    filelinescalc[changecalccellparamline+1]=filelinespos[pullinitcelldmline+1+1]    
    filelinescalc[changecalccellparamline+2]=filelinespos[pullinitcelldmline+1+2]   
    
 




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


