#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <string>
#include <limits>
#include <climits>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

class sparsified_ECP
{
public:
    sparsified_ECP(int x_step, int y_step, double xmin, double xmax, double ymin, double ymax, const vector<tuple<double,double,int>>& data)
        : x_step(x_step), y_step(y_step), xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax)
    {
        sparsified = vector<vector<int>>(x_step, vector<int>(y_step, 0));  // size: x_step × y_step

        double cell_width = (xmax - xmin) / x_step;
        double cell_height = (ymax - ymin) / y_step;

        for (const auto& item : data) {
            double x = get<0>(item);
            double y = get<1>(item);
            int val = get<2>(item);

            int xg = int((x - xmin) / cell_width);
            int yg = int((y - ymin) / cell_height);

            // Clamp to [0, x_step - 1] and [0, y_step - 1]
            xg = max(0, min(xg, x_step - 1));
            yg = max(0, min(yg, y_step - 1));

            sparsified[xg][yg] += val;
        }
    }

    void extend()
    {
        extended = vector<vector<int>>(x_step, vector<int>(y_step, 0));

        for (int i = 0; i < x_step; ++i) {
            for (int j = 0; j < y_step; ++j) {
                extended[i][j] = sparsified[i][j];
                if (i > 0) extended[i][j] += extended[i-1][j];
                if (j > 0) extended[i][j] += extended[i][j-1];
                if (i > 0 && j > 0) extended[i][j] -= extended[i-1][j-1];
            }
        }
    }

    int x_step, y_step;
    double xmin, xmax, ymin, ymax;
    vector<vector<int>> sparsified;
    vector<vector<int>> extended;
};

double distance(sparsified_ECP& first, sparsified_ECP& second, int p = 1)
{
    if (first.xmin != second.xmin || first.xmax != second.xmax ||
        first.ymin != second.ymin || first.ymax != second.ymax ||
        first.x_step != second.x_step || first.y_step != second.y_step) 
    {
        cerr << "Error: Wrong dimensions of sparsified ECPs!" << endl;
        return 0;
    }

    double cell_width = (first.xmax - first.xmin) / first.x_step;
    double cell_height = (first.ymax - first.ymin) / first.y_step;
    double cell_area = cell_width * cell_height;

    double dist = 0;

    for (int i = 0; i < first.extended.size(); ++i) {
        for (int j = 0; j < first.extended[i].size(); ++j) {
            double diff = abs(first.extended[i][j] - second.extended[i][j]);
            dist += pow(diff * cell_area, p);
        }
    }

    return pow(dist, 1.0 / p);
}

vector<tuple<double, double, int>> readDataFromFile(const string& filename) 
{
    vector<tuple<double, double, int>> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return data;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        double val1, val2;
        int val3;
        if (ss >> val1 >> val2 >> val3) {
            data.emplace_back(val1, val2, val3);
        }
    }

    file.close();
    return data;
}

void findMinMax(const vector<tuple<double, double, int>>& data, double& xmin, double& xmax, double& ymin, double& ymax) 
{
    xmin = numeric_limits<double>::max();
    xmax = numeric_limits<double>::lowest();
    ymin = numeric_limits<double>::max();
    ymax = numeric_limits<double>::lowest();

    for (const auto& item : data) {
        double x = get<0>(item);
        double y = get<1>(item);
        xmin = min(xmin, x);
        xmax = max(xmax, x);
        ymin = min(ymin, y);
        ymax = max(ymax, y);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <file1> <file2>\n";
        return 1;
    }

    std::string file1 = argv[1];
    std::string file2 = argv[2];

    vector<tuple<double, double, int>> data1 = readDataFromFile(file1);
    vector<tuple<double, double, int>> data2 = readDataFromFile(file2);

    // Manually set bounding box
    double xmin = -22;
    double xmax = 22;
    double ymin = -22;
    double ymax = 22;

    int x_step = 10000;
    int y_step = 10000;

    cout << "Combined bounding box: "
         << "xmin = " << xmin << ", xmax = " << xmax
         << ", ymin = " << ymin << ", ymax = " << ymax 
        << ", x_step = " << x_step << ", y_step = " << y_step << endl;

    sparsified_ECP ecp1(x_step, y_step, xmin, xmax, ymin, ymax, data1);
    sparsified_ECP ecp2(x_step, y_step, xmin, xmax, ymin, ymax, data2);

    ecp1.extend();
    ecp2.extend();

    double d = distance(ecp1, ecp2, 1);
    cout << fixed << setprecision(2);  // Set desired precision (e.g., 2 decimals)
    cout << "L1 distance = " << d << endl;

    return 0;
}
