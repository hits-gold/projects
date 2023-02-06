# 항공이미지활용 건물변화탐지
### 한 줄 소개
- 일정 기간 전 후의 특정 좌표 항공이미지를 비교해 건물의 변화 탐지를 한다. pytorch_segmentation과 MMsegmentation을 활용했다.

<프로젝트 기간 : 2022.10.15 ~ 2022.11.07>   
<Tags : CV, Sementic Segmentation>  
<역할 : 선행연구조사, MMsegmentation>

<img src = './img/mm.png'>

- 건물변화 탐지를 위한 input image는 위와 같다.
- 데이터를 받기 전 task 이름만으로 building change detection이란 detection분야에 대한 선행연구 조사를 실시했다.
- 하지만 위 사진처럼 전 후 사진이 가로로 붙어있는 형태에서 Sementic Segmentation을 활용해 픽셀 단위의 분류를 진행해야했다.
- 이를 위해 커스텀한 대회 baseline 코드와 MMsegmentatoin을 활용했다.
- MMsegmentatoin 같은 경우 config 설정을 하는 데만 오랜 시간이 걸렸고, 기한 내에 원하는 결과를 도출하지는 못했다.
- 본 프로젝트를 통해 질 높은 오픈 소스 라이브러리를 잘 다룰 줄도 알아야된다고 느꼈다.
