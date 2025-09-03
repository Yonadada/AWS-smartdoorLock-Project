# train_face_recognition.py
from ultralytics import YOLO
import os


def train_yolo_face():
    """YOLO 얼굴인식 모델 학습"""
    
    # 1. 사전 훈련된 YOLOv8 모델 로드
    model = YOLO('yolov8n.pt')  # nano 버전 (빠름, 가벼움)
    # model = YOLO('yolov8s.pt')  # small 버전 (정확도 좀 더 높음)
    
    # 2. 학습 설정
    results = model.train(
        data='face_data.yaml',
        epochs=50,              
        imgsz=128,             # 더 작게
        batch=2,               # 배치 크기 늘려서 효율성 향상
        name='face_test',
        device='cpu',
        workers=0,             
        cache=True,            # 캐시 활성화로 속도 향상
        amp=False,
        save_period=10,        # 저장 빈도 줄임
        patience=50            # early stopping 방지
)
    
    print("학습 완료!")
    print(f"최고 성능 모델: {model.trainer.best}")
    
    return model

def validate_model(model_path):
    """학습된 모델 검증"""
    model = YOLO(model_path)
    
    # 검증 실행
    metrics = model.val()
    
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    return metrics

def test_prediction(model_path, image_path):
    """학습된 모델로 예측 테스트"""
    model = YOLO(model_path)
    
    # 예측 실행
    results = model(image_path)
    
    # 결과 출력
    for r in results:
        print(f"감지된 얼굴 수: {len(r.boxes)}")
        for box in r.boxes:
            print(f"신뢰도: {box.conf[0]:.3f}")
            print(f"좌표: {box.xyxy[0]}")
    
    # 결과 이미지 저장
    results[0].save('prediction_result.jpg')
    print("결과 이미지가 prediction_result.jpg로 저장되었습니다.")

if __name__ == "__main__":
    print("YOLO 얼굴인식 모델 학습 시작!")
    
    # 필요한 패키지 설치 확인
    try:
        from ultralytics import YOLO
        print("ultralytics 패키지 확인됨")
    except ImportError:
        print("ultralytics 패키지가 없습니다. 설치해주세요:")
        print("pip install ultralytics")
        exit(1)
    
    # 데이터셋 경로 확인 (scripts 폴더에서 상위 폴더 참조)
    if not os.path.exists('face_data.yaml'):
        print("face_data.yaml 파일이 없습니다. 먼저 생성해주세요!")
        exit(1)
    
    if not os.path.exists('../data/datasets/images'):
        print("이미지 폴더를 찾을 수 없습니다.")
        exit(1)
        
    if not os.path.exists('../data/datasets/labels'):
        print("라벨 폴더를 찾을 수 없습니다.")
        exit(1)
    
    # 학습 시작
    model = train_yolo_face()
    
    # 학습 완료 후 최고 성능 모델로 검증
    best_model_path = '../runs/train/face_test/weights/best.pt'
    
    if os.path.exists(best_model_path):
        print("모델 검증 중...")
        validate_model(best_model_path)
        
        # 테스트 이미지가 있다면 예측 테스트
        test_images = ['../data/datasets/images/face1.jpg']  # 첫 번째 이미지로 테스트
        for test_img in test_images:
            if os.path.exists(test_img):
                print(f"{test_img}로 예측 테스트...")
                test_prediction(best_model_path, test_img)
                break
    
    print("\n🎉 모든 과정이 완료되었습니다!")
    print(f"📁 학습 결과: runs/train/face_recognition/")
    print(f"🏆 최고 모델: {best_model_path}")