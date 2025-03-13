pipeline {
    agent any

    environment {
        MLFLOW_TRACKING_URI = "http://mlflow:5000"
    }

    stages {
        stage('Check for New Data') {
            steps {
                script {
                    def newData = sh(script: "ls -1 data/ | wc -l", returnStdout: true).trim()
                    if (newData.toInteger() > 0) {
                        echo "New data detected, proceeding with retraining..."
                    } else {
                        error "No new data found, skipping retraining."
                    }
                }
            }
        }

        stage('Retrain Model') {
            steps {
                sh 'python train.py'
            }
        }

        stage('Evaluate Model') {
            steps {
                script {
                    def prevAccuracy = sh(script: "cat prev_accuracy.txt", returnStdout: true).trim().toFloat()
                    def newAccuracy = sh(script: "python evaluate.py", returnStdout: true).trim().toFloat()

                    echo "Previous Accuracy: ${prevAccuracy}"
                    echo "New Accuracy: ${newAccuracy}"

                    if (newAccuracy > prevAccuracy) {
                        echo "New model performs better, proceeding with deployment."
                        sh "echo ${newAccuracy} > prev_accuracy.txt"
                    } else {
                        error "New model did not improve. Keeping the old model."
                    }
                }
            }
        }

        stage('Register New Model in MLflow') {
            steps {
                sh 'python register_model.py'
            }
        }

        stage('Deploy Model') {
            steps {
                sh 'mlflow models serve -m "models:/IrisModel/Production" -p 5001 --no-conda &'
            }
        }
    }

    post {
        failure {
            echo "Training failed. Investigate the logs."
        }
    }
}
