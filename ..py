import jpype
from konlpy.tag import Okt
import os

# 1. JVM 경로 (설치한 JDK 버전에 따라 다를 수 있음)
jvm_path = r"C:\Program Files\Java\jdk-23\bin\server\jvm.dll"

# 2. okt.jar 파일 경로 (지금 복사한 위치)
jar_path = r"C:\Users\Lenovo\konlpy\java\okt-2.1.0.jar"

# 3. JVM 시작
if not jpype.isJVMStarted():
    jpype.startJVM(jvm_path, "-Djava.class.path=" + jar_path)

# 4. Okt 형태소 분석기 테스트
okt = Okt()
print(okt.morphs("이제 진짜 작동한다! 드디어 끝이다!"))

