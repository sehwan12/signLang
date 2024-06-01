Sign Language Detector
---------
<p>
1.Overview
이 프로그램은 웹캠에서 손 인식을 통해 한국어 수어를 인식해 출력하는 프로그램입니다.<br />
기술 구현을 위해 opencv, mediapipe를 이용하였습니다.<br />
  
![signStudy](https://github.com/sehwan12/signLang/assets/58384653/f6241e2c-486f-4be1-babc-ec61cd0e9f36)

수어 인식 훈련
<br />
![signDetect](https://github.com/sehwan12/signLang/assets/58384653/73468da7-a555-4f43-81f9-dfe8e2f05857)
-훈련된 알고리즘으로 수어 인식 후 텍스트 출력<br />
![signDetect2](https://github.com/sehwan12/signLang/assets/58384653/a86d1a69-a3ad-47d7-bede-5e8597e4d958)
-초성+중성까지 출력한 모습<br />
</p>
<p>
2.배경지식
  
![img](https://github.com/sehwan12/signLang/assets/58384653/56cc5552-a3cb-4d1c-b953-499230fd57aa)

mediapipe에서 손 인식은 손 부분마다 벡터를 나눠 인식한다.<br />

-knn알고리즘<br />
데이터로부터 거리가 가까운 'k'개의 다른 데이터의 레이블을 참조하여 분류하는 알고리즘으로, 거리를 측정할 때 '유클리디안 거리'계산법을 사용한다.<br />
</p>
<p>
3.과정
<br />
1.벡터의 뺄셈 연산을 통해 많은 벡터를 만들어낸다.<br />
2.그 이후 벡터와 벡터사이의 각도를 구한다.<br />
3.signstudy.py에서 a키를 누를 때마다 각각의 손 동작을 인식 후, 벡터를 가져와서 test.txt에 저장<br />
4.임의의 20개줄을 가져와서 dataset.csv에 저장<br />
5.구한 각도를 DetectKNN.py에서 KNN알고리즘을 사용해서 구한 각도가 어떤 제스쳐를 뜻하는지 알아낸다.(이때 dataset.csv)를 사용<br />
#한글/영어 수어 이미지는 sign_language_data에서 확인 가능<br />
</p>
<p>
<h1>
4.한계점
</h1>
<br />1.한글용 인식 프로그램을 만들기 전, 영어용 프로그램을 먼저 만들어봤을 때는 인식률이 좋았으나, 한글의 경우 자음과 모음의 갯수를 합치면 손동작의 수가 더 많아져 인식률이 떨어졌다.<br />
2.특히, 'ㅏ'와 'ㅗ'같이 손가락의 모양은 같고 앞/뒷면만 다른 경우 인식을 못하고 있다.<br />
3.받침 'ㄵ'같은 경우 관련 수어를 찾지 못해 구현하지 못했다.<br />
</p>
---
<p>
5.참조
https://developeralice.tistory.com/12<br />
https://hangeul.naver.com/font(나눔글꼴 다운로드)<br />
</p>
