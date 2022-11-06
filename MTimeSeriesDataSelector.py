import MAbnormalityDetector
import csv
import numpy as np

def getTimeSeriesData(data_time, data_num):
    time_series_data='./sample/time_series_data.csv'

    csv_file=open(time_series_data, 'r', encoding='utf-8')
    rdr=csv.reader(csv_file)
    data=[]
    for d in rdr:
        data.append(d)

    data_np=np.array(data)
    index=np.where(data_np==data_time)
    index_i=index[0][0]

    data_tem=data_np[index_i-data_num:index_i,3]
    data_hum=data_np[index_i-data_num:index_i,4]

    start_time=data_np[index_i-data_num,2]
    end_time=data_np[index_i,2]

    s_1, s_2='{', '}'
    xml=f'''
<mtdl:analysedData xsi:type="maov:TimeSeriesReturnDataType">
    <maov:SensorList>
        <maov:Sensor id="Tem-001" unit="celsius"/>
        <maov:Sensor id="Hum-001" unit="percentage"/>
    </maov:SensorList>
    <maov:StartTime>{start_time}</maov:StartTime>
    <maov:EndTime>{end_time}</maov:EndTime>
    <maov:SampleFrequency>P3H</maov:SampleFrequency>
    <maov:Data>
        {s_1} "sensor01": "Tem-001",
        "data01": {data_tem},
        "sensor02": "Hum-001",
        "data02": {data_hum}
        {s_2}
    </maov:Data>
</mtdl:analysedData>'''

    print(f'---MTimeSeriesDataSelector getTimeSeriesData() return--- {xml} \n\n')



    return xml