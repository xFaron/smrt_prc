from ultralytics import YOLO
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", help="Experiment number. Used to save logs in experiment folder")

    args = parser.parse_args()

    model = YOLO("yolo11n.pt")
    results = model.train(
        data="/home/xfaron/Desktop/Code/Playground/test_construction/configs/roi_yolo.yaml",
        epochs=100,
        imgsz=640,
        name=f"roi_exp_{args.experiment}",
        project="/home/xfaron/Desktop/Code/Playground/test_construction/experiments",
    )
