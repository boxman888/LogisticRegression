/********************************************************************
 * Implements logistic regression analysis in an attempt to learn
 * the weight/hypothesis coefficients for images of hand 
 * drawn 4s and 9s.
 * Created: Jun 26 2020 
 * Author: Ethan Patterson
 *******************************************************************/

#include <iostream>
#include <fstream>
#include <cstdlib>

#include <Eigen/Dense> // Linear Algebra library. Don't forget to add path to Makefile.
//#include <unsupported/Eigen/MatrixFunctions>

#include "USPS.h" // For reading data files.

#ifndef ITR
#define ITR 100
#endif

#ifndef ALPHA
#define ALPHA 0.1
#endif

// Made for convinance of managing 
// the max and min values of the data.
struct Tuple{
    float x;
    float y;
};

void loadData(Eigen::MatrixXf &X, Eigen::VectorXf &Y, std::ifstream &fp){
    int r = 0;
    USPS data;
    // Read data line by line and sotre it into X matrix.
    while (fp >> data){
        for (int i = 0; i < COL_SIZE-1; i++){
            X(r, i) = std::stof(data.feature[i], nullptr);
            Y(r) = std::stof(data.feature[COL_SIZE-1]); // Values of interest.
        }
        r++;
    }

    fp.clear();
    fp.seekg(0, fp.beg);// Reset to head of file. 
}
/**
 * @param fp is a file pointer.
 * Function gets the number of rows in a file.
 */
int getFileRowCount(std::ifstream &fp){
    int num_of_rows = 0;
    std::string rows;
    while (std::getline( fp, rows ))
        num_of_rows++;
    
    fp.clear();
    fp.seekg(0, fp.beg); // Reset to head of file.
    return num_of_rows;
}
/**
* @param W // Vector of estimated coefficients.
* @param Xi // Single row of the X matrix.
* Apply the sigmoid function to the linear regression equation
* for a single row of the X matrix. W^TX_i = W_0 + W1_1*X_1 + ...
*/
float Sigmoid(Eigen::VectorXf &W, Eigen::VectorXf &Xi){
    //return 1.0 / (1.0 + (-1 * W.transpose() * Xi).exp());
    Eigen::VectorXf e (0);
    e  = (-1 * W.transpose() * Xi).array().exp();
    return 1.0 / (1.0 + e(0));
}
/**
 * @param W Vector of estimated coefficients.
 * @param Y Vector of observed variables. 
 * @param X Matrix of regressor variables.
 * @param Delta Change in gradient slope.
 * Updates the gradient delta of system. This update is done by
 * updating the Delta by reference. This was done as I found it simpler to work with.
 */
void Gradient(Eigen::VectorXf &W, Eigen::VectorXf &Y, Eigen::MatrixXf &X, Eigen::VectorXf &Delta){
    Eigen::VectorXf Xi ( X.rows() );
    float hypothosis = 0.0;

    for (int i = 0; i < X.rows(); i++){
        Xi = X.row(i);
        hypothosis = round(Sigmoid(W, Xi));
        Delta = Delta + (hypothosis - Y(i)) * Xi;
    }
}
/**
 * @param X Matrix of regressor variables.
 * @param W Vector of estimated coefficients.
 * @param Y Vector of observed variables.
 * Function to evaluate the current accuracy of the evaluations.
 * Returns current accuracy.
 */ 
float Cost(Eigen::MatrixXf &X, Eigen::VectorXf &W, Eigen::VectorXf &Y){
    Eigen::VectorXf Xi ( X.rows() );
    float hypothosis = 0.0;
    
    float count = 0.0;
    for (int i = 0; i < X.rows(); i++){
        Xi = X.row(i);
        hypothosis = round(Sigmoid(W, Xi));
        
        if (hypothosis == (float)Y(i)){
            count++;
        }
    }
    return count / Y.size() * 100.0;
}
/**
 * @param X Matrix of Quantitative Variables.
 * @param W Learned vector of weight coefficients, hypothosis.
 * @param Y Vector of Response Variables.
 * @param alpha The learning rate to apply.
 * @param iter The number of iterations to complete.
 * Use batch gradient descent to minimize the cost/error of the system.
 * Returns the cost history of the system over iterations.
 */
float* BGD(Eigen::MatrixXf &X, Eigen::VectorXf &W, Eigen::VectorXf &Y, float alpha, int itr, const Tuple &maxmin){
    Eigen::VectorXf Delta ( X.cols() );
    float* costHistory = new float[itr];
    Delta = Eigen::VectorXf::Ones( X.cols() );

    float cost = 0.0;
    for (int i = 0; i < itr; i++){
        Gradient(W, Y, X, Delta);

        W = W - alpha * Delta;
        cost = Cost(X, W, Y);
        costHistory[i] = cost;
    }
    return costHistory;
}

/**
 * @param X Matrix of Quantitative Variables.
 * @param Y Vector of Response Variables.
 * Nomralize the data between 0 and 1.
 * Returns a tuple of the max and min value.
 * Max is the x component and min is the y.
 */
Tuple normalize(Eigen::MatrixXf &X, Eigen::VectorXf &Y){
    float max_x, max_y, max;
    float min_x, min_y, min;
    Tuple maxmin;

    Eigen::MatrixXf::Index maxRow,maxCol;
    Eigen::MatrixXf::Index minRow,minCol;

    max_x = X.maxCoeff(&maxRow, &maxCol);
    max_y = Y.maxCoeff(&maxRow, &maxCol);

    min_x = X.minCoeff(&minRow, &minCol);
    min_y = Y.minCoeff(&minRow, &minCol);
    
    if (max_x < max_y)
        max = max_y;
    else
        max = max_x;

    if (min_x < min_y)
        min = min_x;
    else
        min = min_y;

    for (int i = 0; i < X.rows(); i++){
        for (int j = 0; j < X.cols(); j++){
            X(i,j) = (X(i,j) - min) / (max - min);
        }
    }
    
    for (int i = 0; i < Y.rows(); i++)
        Y(i) = (Y(i) - min) / (max - min);
        
    maxmin.x = max;
    maxmin.y = min;

    return maxmin;
}
/**
 * @param cost Dynamic array of cost history.
 * @param W Learned vector of weight coefficients, hypothesis.
 * @param Y Vector of Response Variables.
 * Saves data produced such as the cost history, weights/hypotheses, and normalization min max.
 */
void saveData(float *cost, const Tuple &maxmin, Eigen::VectorXf &W){
    std::ofstream weights("weights.csv");
    std::ofstream norm("normalize.csv");
    std::ofstream costHistory("cost_history.csv");

    if (weights.is_open()){
        for (int i = 0; i < W.size()-1; i++)
            weights << W(i) << ",";
        weights << W(W.size()-1);
    }
    weights.close();

    if (norm.is_open()){
        norm << "max" << "," << maxmin.x << "\n";
        norm << "min" << "," << maxmin.y << "\n";  
    }
    norm.close();

    if(costHistory.is_open()){
        for (int i = 0; i < ITR; i++)
            costHistory << i << "," << cost[i] << "\n";
    }
    costHistory.close();
}

int main()
{
    // Files in training and testing data.
    std::ifstream fp_train("./data/data/usps-4-9-train.csv");
    std::ifstream fp_test("./data/data/usps-4-9-test-shuf.csv");

    // Build maxtix for traning and testing data.
    Eigen::MatrixXf X(getFileRowCount(fp_train), COL_SIZE);
    Eigen::MatrixXf X_test(getFileRowCount(fp_test), COL_SIZE);
    
    // Build Vectors for target variables and weight/hypotheses.
    Eigen::VectorXf Y( X.rows() );
    Eigen::VectorXf Y_test( X_test.rows() );
    Eigen::VectorXf W ( X.cols() );// Size is the number of features/dimensions.

    Tuple maxmin; // Used to keep track of max and min value during normalization.
    float *costHistory; // Pointer for array to cost history.

    // Load data into matrices and vectors.
    loadData(X, Y, fp_train);
    loadData(X_test, Y_test, fp_test);
    fp_train.close(); // Don't need this anymore, data is now in the matrix.
    fp_test.close();

    // Set all initial weight/hypothesis values.
    W = Eigen::VectorXf::Random( X.cols(), 1 );
    
    maxmin = normalize(X, Y); // Normalize data.
    normalize(X_test, Y_test); // Keep max and min values for future data sets.
    std::cout << "Max: " << maxmin.x << " Min: " << maxmin.y << std::endl;

    // Run batch gradient descent.
    costHistory = BGD(X, W, Y, ALPHA, ITR, maxmin);
    std::cout << "test score: " << Cost(X_test, W, Y_test) << std::endl;
    
    saveData(costHistory, maxmin, W);
    delete costHistory;

    return 0;
}