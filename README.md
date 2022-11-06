# TimeSeries

#### 시계열 데이터 관리 및 이상 감지 소프트웨어 결과는 다음과 같다.

<img width="427" alt="image" src="https://user-images.githubusercontent.com/117606355/200201407-1e68a54a-fd6f-4836-b8e9-c797fe6e2a08.png">

<img width="427" alt="image" src="https://user-images.githubusercontent.com/117606355/200201410-8e793437-ba48-44b1-8035-120def00a313.png">

<img width="427" alt="image" src="https://user-images.githubusercontent.com/117606355/200201411-008b1f1e-62ca-435f-8285-c59d10b2b22c.png">


먼저 이상 감지 미션을 시작 후 MAbnormalityAnalyser는 MAbnormalityDetector에게 이상 탐지를 수행하라는 getDetectedAbnoality()를 호출한다. 
그 후 MAbnormalityDetector는 이상 탐지를 위해 MCamera에 getVideoURL을 요청하고, MCamera는 video_url을 리턴한다. 
그 후 MAbnormalityDetector는 영상의 프레임을 하나씩 불러와 Yolo v5를 통해 객체 탐지를 수행한다. 
이 때 MAbnormalityDetector는 객체 탐지 결과를 기반으로 특정 관리 대상에 대하여 이상을 탐지한다.
본 용역 소프트웨어에서 MAbnormalityDetector가 이상을 탐지하기 위해서는 아래의 조건을 모두 만족하여야한다.
-	관리 대상이 n 프레임 연속으로 탐지(본 예제에서는 n=10)
-	관리 대상이 area 크기 이상으로 탐지(본 예제에서는 area=130000)
그 후MAbnormalityDetector는 이상 탐지 분석 결과를 MPEG-IoMT 표준 형식을 따르는 xml 형태로 리턴한다. 해당 xml 내에는 이상 탐지 시간(DetectedTime), 이상 탐지 유무(AbnormalityDetection), 이상 탐지영상(DetectedVideoURL)이 포함되어 있다. 

MAbnormalityAnalyser는 MAbnormalityDetector으로부터 분석 결과를 기반으로 이상 상황을 분석한다. 
이상 상황을 면밀히 분석하기 위해서 시계열 센서 데이터가 추가로 필요하다. 
때문에 MAbnormalityAnalyser는 MTimeSeriesDataSelector에서 getTimeSeriesData()을 호출하여 MAbnormalityDetector에서 받은 이상 탐지 시간(DetectedTime)으로부터 과거 data_num개의 데이터를 가져올 것을 요청한다. 
MTimeSeriesDataSelector는 이상 탐지 시간(DetectedTime)과 data_num(패킹할 시계열 데이터 수)를 기반으로 CSV 파일로부터 시계열 데이터인 온도, 습도 센서 정보를 MPEG-IoMT 표준 형식을 따르는 xml 형태로 리턴한다.
해당 xml에는 패킹 된 센서 정보(Sensor의 id, unit), 데이터 추출 관련 시간 정보(StartTime, EndTime), 데이터 추출 간격(SampleFrequency), 패킹된 데이터(Data)가 포함되어 있다.

MAbnormalityAnalyser는 두개의 분석기(MAbnormalityDetector, MTimeSeriesDataSelector)로부터 받은 분석 결과를 기반으로 이상 상황을 분석하고 저장하거나 해당 이상 상황을 해결하기 위하여 MActuator에게 명령을 보낸다.
