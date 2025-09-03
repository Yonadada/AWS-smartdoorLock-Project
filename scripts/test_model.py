# test_model.py
from ultralytics import YOLO
import cv2
import os

# 현재 위치 확인
print("현재 작업 디렉토리:", os.getcwd())

# 경로들 하나씩 확인
paths_to_check = [
    './runs/',
    './runs/detect/',
    './runs/detect/face_test/',
    './runs/detect/face_test/weights/',
    './runs/detect/face_test/weights/best.pt'
]

for path in paths_to_check:
    exists = os.path.exists(path)
    print(f"{path} 존재: {exists}")
    if exists and os.path.isdir(path):
        try:
            contents = os.listdir(path)
            print(f"  내용: {contents}")
        except:
            print("  내용을 읽을 수 없음")

def test_trained_model():
    """학습된 YOLO 모델 테스트"""
    
    # 가능한 모델 경로들 확인 (최신 것부터)
    possible_model_paths = [
        './runs/detect/face_test/weights/best.pt',  # detect 폴더로 변경
        './runs/detect/face_test/weights/last.pt'
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"모델 발견: {path}")
            break
    
    if not model_path:
        print("학습된 모델을 찾을 수 없습니다.")
        print("먼저 train_face.py를 실행해서 모델을 학습시켜주세요.")
        return
    
    # 모델 로드
    model = YOLO(model_path)
    print(f"모델 로드 완료: {model_path}")
    
    # 테스트할 이미지들
    test_images = [
        '../data/datasets/images/face1.jpg',
        '../data/datasets/images/face2.jpg',
        '../data/datasets/images/face3.jpg'
    ]
    
    for i, image_path in enumerate(test_images):
        if not os.path.exists(image_path):
            print(f"이미지를 찾을 수 없습니다: {image_path}")
            continue
            
        print(f"\n=== 테스트 {i+1}: {image_path} ===")
        
        # 예측 실행
        results = model(image_path)
        
        # 결과 출력
        for r in results:
            if len(r.boxes) > 0:
                print(f"감지된 얼굴 수: {len(r.boxes)}")
                for j, box in enumerate(r.boxes):
                    confidence = box.conf[0].item()
                    coordinates = box.xyxy[0].tolist()
                    print(f"  얼굴 {j+1}:")
                    print(f"    신뢰도: {confidence:.3f}")
                    print(f"    좌표: x1={coordinates[0]:.1f}, y1={coordinates[1]:.1f}, x2={coordinates[2]:.1f}, y2={coordinates[3]:.1f}")
            else:
                print("얼굴이 감지되지 않았습니다.")
            
            # 결과 이미지 저장
            result_filename = f'test_result_{i+1}.jpg'
            r.save(result_filename)
            print(f"결과 저장: {result_filename}")

def test_webcam():
    """웹캠으로 실시간 얼굴 감지 테스트"""
    
    # 모델 경로 찾기
    possible_model_paths = [
        './runs/detect/face_test/weights/best.pt',  # detect로 변경
        './runs/detect/face_test/weights/last.pt'   # detect로 변경
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("모델 파일을 찾을 수 없습니다.")
        return
    
    model = YOLO(model_path)
    print(f"웹캠 테스트 시작 (모델: {model_path})")
    print("'q' 키를 누르면 종료됩니다.")
    
    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break
        
        # YOLO 예측
        results = model(frame, verbose=False)  # verbose=False로 출력 줄임
        
        # 결과를 프레임에 그리기
        annotated_frame = results[0].plot()
        
        # 감지된 얼굴 수 표시
        face_count = len(results[0].boxes) if results[0].boxes else 0
        cv2.putText(annotated_frame, f"Faces: {face_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection Test', annotated_frame)
        
        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("웹캠 테스트 종료")

if __name__ == "__main__":
    print("YOLO 얼굴인식 모델 테스트")
    print("1. 이미지 파일 테스트")
    print("2. 웹캠 실시간 테스트")
    
    choice = input("선택하세요 (1 또는 2): ")
    
    if choice == "1":
        test_trained_model()
    elif choice == "2":
        test_webcam()
    else:
        print("1 또는 2를 입력해주세요.")
        test_trained_model()  # 기본값으로 이미지 테스트