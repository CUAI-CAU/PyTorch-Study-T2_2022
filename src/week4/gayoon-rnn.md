# 시퀀스 배열로 다루는 순환 신경망   
* 작성자 : 김가윤   
* 참고 : 모두의 딥러닝 개정판   
   
## 순환 신경망   
* 과거에 입력된 데이터와 나중에 입력된 데이터 사이의 관계를 고려해 고안된 신경망    
* 여러 개의 데이터가 순서대로 입력되었을 때 앞서 입력받은 데이터를 잠시 기억   
* 기억된 데이터가 얼마나 중요한지를 판단하여 별도의 가중치 부여, 같은 층에서 맴도는 성질 때문에 순환이라고!   

![image](https://thebook.io/img/080228/260.jpg)   

## LSTM (Long Short Term Memory)   
* 기울기 소실 문제 보완 방법, 반복 전에 다음 층으로 기억된 값 넘길지 여부 관리 단계를 추가      
![image](https://thebook.io/img/080228/262_1.jpg)   
