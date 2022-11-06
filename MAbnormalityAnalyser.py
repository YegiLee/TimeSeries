import MAbnormalityDetector
import MTimeSeriesDataSelector

def getAnalysedAbnormality():

    print("+++++++++ MAbnormalirtAnalyser requests getDetectedAbnormality() to MAbnormalityDetector. +++++++++\n")
    analysedDataFromMAD=MAbnormalityDetector.getDetectedAbnormality()

    print("+++++++++ MAbnormalirtAnalyser requests getTimeSeriesData() to MTimeSeriesDataSelector. +++++++++\n")
    analysedDataFromTSDS=MTimeSeriesDataSelector.getTimeSeriesData(analysedDataFromMAD[1], data_num=30)

    print(f'analysing abnormal situation from MCamera and Time Series Data...\n')

    xml='''                            
<mtdl:analysedData xsi:type="maov:AnalysedLivestockAbnormalityType">
    <maov:Livestock>urn:mpeg:mpeg-IoMT:01-LivestockCS-NS:PIG </maov:Livestock>
    <maov:AbnormalityType>Reek</maov:AbnormalityCause>
    <maov:ExpectedMPEGVAction xsi:type="sdcv:CoolingType" id="cooling01" activate="true" intensity="40" />
</mtdl:analysedData>
    '''
    return xml