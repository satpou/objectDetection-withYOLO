from app import app, create_app

app = create_app()

if __name__ == '__main__':
    from app.services.detector import init_model
    from app.services.gesture import init_landmarker
    from app.services.camera import init_camera
    init_model()
    init_landmarker()
    init_camera()
    print("[INFO] Server ready at http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)