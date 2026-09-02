import numpy as np

fileendlist=['baxis16radp995.sh','baxis16radp99.sh','baxis16radp98.sh','baxis16radp97.sh','baxis16radp96.sh']
filestartlist=['relaxbase.sh']

filecoordlist=['../../../hydro/brav/results/hydro16.out']



pressurelist=['160.0,']*len(fileendlist)

axisfactor=[0.995, 0.99,0.98,0.97,0.96 ]
axis='B'



def normvec(vec):
	return vec/np.sqrt(np.dot(vec,vec))


for z in range(len(fileendlist)):
    fileorigcalc=filestartlist[0]
    fileoriginitbase=filecoordlist[0]

    
    filenew=fileendlist[z]

    fileoriginit=fileoriginitbase
#loads proper files
    print(filenew)
    print(fileoriginit)
    fnew=open(filenew,"w")
    foldinit=open(fileoriginit,"r")
    foldcalc=open(fileorigcalc,"r")
    filelinespos=foldinit.readlines()
    filelinescalc=foldcalc.readlines()
    
   #fixes pressure 
    pressuretrigger='  press='
    changepressline=-10
    for i in range(len(filelinescalc)):
        if filelinescalc[i][0:len(pressuretrigger)]==pressuretrigger:
            changepressline=i
            break       
    newpressline='  press='+pressurelist[z]+'\n'
    filelinescalc[changepressline]=newpressline



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
    
    latticemat=np.zeros([3,3])
    varstart=np.zeros([3,1])
    varend=np.zeros([3,1])
#finds lattice params 
    for j in range(3):
        linecurrent=filelinespos[pullinitcelldmline+1+j]
        numeric=0

        for i in range(len(linecurrent)): 

            if linecurrent[i].isnumeric() and numeric==0:
                numeric=1
                varstart[0]=i-1
            if linecurrent[i]==' ' and numeric==1:
                numeric=2
                varend[0]=i
            if linecurrent[i]!=' ' and numeric==2:
                numeric=3
                varstart[1]=i
            if linecurrent[i]==' ' and numeric==3:
                numeric=4
                varend[1]=i
            if linecurrent[i]!=' ' and numeric==4:
                numeric=5
                varstart[2]=i  
                varend[2]=len(linecurrent)
        for k in range(3):
            latticemat[j,k]=float(linecurrent[int(varstart[k]):int(varend[k])])




	
#finds where to put in the alat    
    calccelldmtrigger='&SYSTEM'
    changecalccelldmline=-10
    for i in range(len(filelinescalc)):
        if filelinescalc[i][0:len(calccelldmtrigger)]==calccelldmtrigger:
            changecalccelldmline=i+6
            break
    #Changes alat
    filelinescalc[changecalccelldmline]="  celldm(1) ="+f"{alat}, \n"

#Finds where to put atompos     	
    calccellatompostrigger= 'ATOMIC_POSITIONS crystal'
    changecalcatomposline=-10
    for i in range(len(filelinescalc)):
        if filelinescalc[i][0:len(calccellatompostrigger)]==calccellatompostrigger:
            changecalcatomposline=i+1
            break        
            
#places atompos                
    for i in range(24):
        filelinescalc[changecalcatomposline+i]=filelinespos[pullinitcellatomposline+i]

#Finds celldofree line 
    celldofreetrigger='  cell_dofree='
    celldofreeline=-10	
    for i in range(len(filelinescalc)):
        if filelinescalc[i][0:len(celldofreetrigger)]==celldofreetrigger:
            celldofreeline=i
            break
 #setts celldofree for job type 
    if axis=='A':
    	filelinescalc[celldofreeline]="  cell_dofree='lockA',\n"
    if axis=='B':
    	filelinescalc[celldofreeline]="  cell_dofree='lockB',\n"
    if axis=='BC':
    	filelinescalc[celldofreeline]="  cell_dofree='BCaxis',\n"

#Finds all the places where job name needs to change 
    filename1trigger='cat >'
    filename2trigger='  prefix ='    
    filename3trigger='$PW_COMMAND <'
    changefileline1=-10
    changefileline2=-10
    changefileline3=-10

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

#Changes job name where needed 
    filelinescalc[changefileline1]=newfilename1line  
    filelinescalc[changefileline2]=newfilename2line  
    filelinescalc[changefileline3]=newfilename3line  


 #Saves lattice vectors for easy use    
    caxis=np.transpose(latticemat[2,:])
    baxis=np.transpose(latticemat[1,:])
    aaxis=np.transpose(latticemat[0,:])

    if axis=='B': 
	    caxisfinal=caxis
	    baxisfinal=baxis*axisfactor[z]
	    aaxisfinal=aaxis
    if axis=='A': 
	    caxisfinal=caxis
	    baxisfinal=baxis
	    aaxisfinal=aaxis*axisfactor[z]
#Makes the final vector list and shrinks the z axis     
    cstrfinal=f"{caxisfinal[0]}"+"   "+f"{caxisfinal[1]}"+"   "+f"{caxisfinal[2]}"+"\n"
    bstrfinal=f"{baxisfinal[0]}"+"   "+f"{baxisfinal[1]}"+"   "+f"{baxisfinal[2]}"+"\n"
    astrfinal=f"{aaxisfinal[0]}"+"   "+f"{aaxisfinal[1]}"+"   "+f"{aaxisfinal[2]}"+"\n"  

    calccellparamline="CELL_PARAMETERS alat"
    changecalccellparamline=-10
    for i in range(len(filelinescalc)):
        if filelinescalc[i][0:len(calccellparamline)]==calccellparamline:
            changecalccellparamline=i+1
            break
    #Changes vectors
    filelinescalc[changecalccellparamline]=astrfinal
    filelinescalc[changecalccellparamline+1]=bstrfinal    
    filelinescalc[changecalccellparamline+2]=cstrfinal


#writes the new file
    fnew.writelines(filelinescalc)
    print('done')
