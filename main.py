from minecraft_obj_detection import minecraft_obj_detection

def main():
    print("-Minecraft Mob Detection-\n")

    obj_detection = minecraft_obj_detection(model_path='./yolov8_minecraft_weights.pt')

    model = input("Real-time detection [Enter any key]\nVideo detection [Enter 'v']\n")
    
    if model == 'v':
        print('\n----------\n')
        video_path = input('Please enter the Video path:\n')
        output_path = input('Please enter the Output video path and name (Example: output.mp4):\n')
        
        # display_detections = input('Display real-time results while detecting? [y/n]: ') == 'y'
        
        obj_detection.video_detection(video_path, output_path)
    else:
        obj_detection.rt_detection()

if __name__ == "__main__":
    main()