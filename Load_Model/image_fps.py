import datetime
import os
from tqdm import tqdm
import numpy as np
import re

def datestdtojd (stddate):
    fmt='%Y-%m-%d'
    stddate = str(stddate[0])+'-'+str(stddate[1])+'-'+str(stddate[2])
    sdtdate = datetime.datetime.strptime(stddate, fmt)
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    return(jdate)

def year_month_day(content,pattern):
        
    Regex = re.compile(pattern)
    
    y_m_d = Regex.search(content) # year_month_day
    if not y_m_d:
        y_m_d=[]
    else :
        y_m_d = y_m_d.groupdict()
        y_m_d = [int(y_m_d['yyyy']),int(y_m_d['mm']),int(y_m_d['dd'])]
    return y_m_d
    

def check_date_pattern(dataset_path, pattern):    
      
    assert pattern.find('yyyy')!=-1, "date pattern is not correct, 'yyyy' is missing"
    assert pattern.find('mm')!=-1, "date pattern is not correct, 'mm' is missing"
    assert pattern.find('dd')!=-1, "date pattern is not correct, 'dd' is missing"
    
    count=0
    
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for file in [f for f in filenames if (f.endswith('.jpg') or f.endswith('.JPG'))]:
            y_m_d = year_month_day(file,pattern)
            
            assert y_m_d, " Date pattern is not matching with file: "+ dirpath+'/'+file
            
            year,month,day = y_m_d
            
            time = datetime.datetime.now()
            assert ( year>=1900 and month>0 and day>0 and y_m_d <= [time.year,time.month,time.day])," Date pattern is not matching with file: "+ dirpath+'/'+file

            try:
                datetime.datetime(year,month,day)
            except:
                raise Exception(" Date pattern is not matching with file: "+ dirpath+'/'+file)

            count+=1
        
    assert count>0, "date pattern is not matching with any filename"
    return 1
    


def get_image_fps(dataset_path, date_pattern, required_rois):

    date_pattern=date_pattern.replace('*','.*')
    date_pattern=date_pattern.replace('yyyy','(?P<yyyy>\d{4})')
    date_pattern=date_pattern.replace('mm','(?P<mm>\d{2})')
    date_pattern=date_pattern.replace('dd','(?P<dd>\d{2})')
    date_pattern='^'+date_pattern+'$'

    check_date_pattern(dataset_path,date_pattern)

    images_fps=[]
    DoY=[]
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for file in [f for f in filenames if(f.endswith('.JPG') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.JPEG'))]:
            src=dirpath + '/' + file
            images_fps.append(src)
            y_m_d = year_month_day(file,date_pattern)
            jullion = datestdtojd(y_m_d) 
            for _ in range(required_rois):
                DoY.append(jullion)
    DoY=np.array(DoY)
    return DoY,images_fps