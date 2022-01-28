# 처음 배우는 딥러닝 챗봇

본 저장소는 "처음 배우는 딥러닝 챗봇" 의 예제 코드를 공유하는 저장소입니다.

원본 저장소 github 주소 :  https://github.com/keiraydev/chatbot

모델 참고 저장소 github 주소 :  https://github.com/hyunwoongko/kochat

# 챗봇 코드 변경하기

- db 변경

sqlite file db로 서버구동 없이 돌아가도록 변경

- tensroflow -> pytorch

모델 부분만 pytorch로 변경

# 실행 방법

- db 생성

```buildoutcfg
train_tools/qna 디렉토리
create_train_data_table.py 실행
load_train_data.py 실행
```