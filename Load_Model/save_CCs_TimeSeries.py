import xlsxwriter
import os
def save_CCs_TimeSeries(folder_location,DoY,RCC, GCC, BCC,pRCC,pGCC,pBCC):


    if (folder_location[-5:] !='.xlsx'):

        assert os.path.exists(folder_location), "folder location does not exist"
        
        folder_location += "/TimeSeries_Results.xlsx"

    workbook  = xlsxwriter.Workbook(folder_location)

    worksheet1 = workbook.add_worksheet('Chronology coefficients')
    worksheet1.write(0, 0, "DoY")
    worksheet1.write_column('A2',DoY)
    worksheet1.write(0, 1, "GCC")
    worksheet1.write_column('B2',GCC)
    worksheet1.write(0, 2, "RCC")
    worksheet1.write_column('C2',RCC)
    worksheet1.write(0, 3, "BCC")
    worksheet1.write_column('D2',BCC)
    
    worksheet2 = workbook.add_worksheet('Phenology Parameters')
    worksheet2.write(0, 0, "Parameters")
    worksheet2.write(1, 0, "Minmum CC")
    worksheet2.write(2, 0, "Maximum CC")
    worksheet2.write(3, 0, "Slope1")
    worksheet2.write(4, 0, "SOS")
    worksheet2.write(5, 0, "Slope2")
    worksheet2.write(6, 0, "EoS")

    worksheet2.write(0, 1, "GCC Parameters")
    worksheet2.write_column('B2',pGCC)
    worksheet2.write(0, 2, "RCC Parameters")
    worksheet2.write_column('C2',pRCC)
    worksheet2.write(0, 3, "BCC Parameters")
    worksheet2.write_column('D2',pBCC)

    workbook.close()