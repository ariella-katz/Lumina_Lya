#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>

// x^2 2F2(1, 1; 3/2, 2, -x^2) / sqrt(pi) using a piecewise approximation
// where 2F2(a, b; c, d; x) is the generalized hypergeometric function
double hypergeo_approx(double x) {
  if (x < 0.) x = -x;

  if (x > 20.) {
    // Region 4: (x > 20) Asymptotic expansion
    const double z = 1. / x;
    const double z2 = z * z;
    return 0.55389595193643551 + 0.56418958354775629 * log(x) +
      exp(-x * x) * z * (0.5 + z2 * (-0.25 + z2 * (0.375 + z2 * (-0.9375 + z2 * (3.28125 - 14.765625 * z2))))) +
      z2 * (-0.14104739588693907 + z2 * (-0.1057855469152043 + z2 * (-0.17630924485867384 +
      z2 * (-0.46281176775401883 + z2 * (-1.6661223639144678 - 7.6363941679413107 * z2)))));
  } else if (x > 5.15) {
    // Region 3: (5.15 < x < 20) Minimax rational approximation
    return
      (0.43682128011845283 + x * (1.1350393218808412 + x * (-2.6886477055342774 +
       x * (0.38197597791453486 + x * (0.50175054988846498 + x * (-0.038945006088876756 +
       x * (-0.01960980532428994 + x * (-0.0012352801187522119 +
       x * (-0.000021929464819640861 - 8.9740333784715440e-8 * x))))))))) /
      (1. + x * (-1.4655446738737176 + x * (-1.4032687362246785 + x * (0.88017831100711925 +
       x * (0.20315464152004808 + x * (-0.050541821629539143 + x * (-0.010129700645083951 +
       x * (-0.00047171453699007742 + x * (-6.5774668524211879e-6 +
       x * (-2.0308831722073883e-8 + 3.0992253950009977e-12 * x))))))))));
  } else if (x > 2.14) {
    // Region 2: (2.14 < x < 5.15) Minimax rational approximation
    return
      (-0.042894679076630557 + x * (0.22824496819198929 + x * (0.025188478146879472 +
       x * (-0.073680556438870441 + x * (0.015478791578754775 + x * (0.03578546465825914 +
       x * (-0.027136284977882523 + x * (0.00988058128216986 + x * (-0.0018266571476397632 +
       x * (0.00015895087049937769 + 2.2012003108278176e-6 * x)))))))))) /
      (1. + x * (-1.4228913485269134 + x * (1.422586587092723 + x * (-0.90352979918845972 +
       x * (0.40905729430981564 + x * (-0.11897768600690815 + x * (0.020260960808591361 +
       x * (-0.00051357699758117836 + x * (-0.00042088721875418447 +
       x * (0.000070884265471591177 + 4.4276369978816760e-7 * x))))))))));
  } else {
    // Region 1: (0 < x < 2.14) Padé approximant at x = 0
    const double x2 = x * x;
    return
      (x2 * (0.56418958354775629 + x2 * (0.085729533651389712 + x2 * (0.022086832488217919 +
       x2 * (0.0017036307776357589 + x2 * (0.00018771767997732845 + x2 * (8.5231641525552708e-6 +
       x2 * (5.0427978836558623e-7 + x2 * (1.3331831289246041e-8 + x2 * (4.1502346771379203e-10 +
       x2 * (5.1601562894552471e-12 + 5.7441158232281185e-14 * x2))))))))))) /
      (1. + x2 * (0.48528497539007305 + x2 * (0.11202066087251571 + x2 * (0.016271004013903894 +
       x2 * (0.0016562768063564546 + x2 * (0.00012438873114514399 + x2 * (7.0533651782670942e-6 +
       x2 * (3.0296362830876420e-7 + x2 * (9.6956364133569300e-9 + x2 * (2.2108600436326525e-10 +
       x2 * (3.2481690234488706e-12 + 2.3420543014955824e-14 * x2)))))))))));
  }
}

double hypergeo_approx_single(double x) {
  if (x < 0.) x = -x;

  if (x > 5.75) {
    // Region 3: (x > 5.75) Asymptotic expansion
    const double z = 1. / x;
    const double z2 = z * z;
    return 0.55389595 + 0.56418958 * log(x) +
      exp(-x * x) * z * (0.5 - z2 * (0.25 - z2 * (0.375 - 0.9375 * z2))) -
      z2 * (0.1410474 + z2 * (0.10578555 + z2 * (0.17630924 + 0.46281177 * z2)));
  } else if (x < 2.01) {
    // Region 1: (0 < x < 2.01) Padé approximant at x = 0
    const double x2 = x * x;
    return
      (x2 * (0.56418958 + x2 * (0.08767343 + x2 * (0.019192468 + x2 * (0.0013059543 + x2 * (0.000085663939 + 2.1347118e-6 * x2)))))) /
      (1. + x2 * (0.48873044 + x2 * (0.10803902 + x2 * (0.013932664 + x2 * (0.0011154961 + x2 * (0.000053149736 + 1.1933712e-6 * x2))))));
  } else {
    // Region 2: (2.01 < x < 5.75) Minimax rational approximation
    return
      (-1.0703712 + x * (2.8182998 + x * (-2.0188538 + x * (0.64780614 - 0.047794327 * x)))) /
      (1. + x * (0.17939221 + x * (-0.67372602 + x * (0.37089533 + x * (-0.041412427 + x * (0.0014120009 - 0.000036370895 * x))))));
  }
}

// x^2 2F2(1, 1; 3/2, 2, -x^2) using a piecewise approximation
// where 2F2(a, b; c, d; x) is the generalized hypergeometric function
double old_hypergeo_approx(double x) {
    if (x < 0) x = -x;

    if (x < 1.0) {
        // Region 1: (0 < x < 1) Padé approximant at x = 0
        double x2 = x * x;
        return
        ((((((0.000080548582624363828 * x2 + 0.0014065242642412613) * x2
            + 0.029495703042659217) * x2 + 0.13200462851048492) * x2
            + 1.0) * x2))
        /
        ((((((0.000021427151329799372 * x2 + 0.00071524102170766482) * x2
            + 0.010997258292979713) * x2 + 0.095719468101709747) * x2
            + 0.46533796184381826) * x2 + 1.0));
    }
    else if (x < 2.7) {
        // Region 2: (1 < x < 2.7) Minimax rational approximation
        return
        ((((((0.010801624381328885 * x - 0.076770159937328218) * x
            + 0.36177108816499369) * x - 0.77092840594516447) * x
            + 1.0350089217815957) * x - 0.010135663080884468) * x + 0.0013073784373521217)
        /
        ((((((0.0028590058483104009 * x - 0.016951456661921077) * x
            + 0.093300759774911341) * x - 0.23934329525337882) * x
            + 0.60599457869362251) * x - 0.70062864872271075) * x + 1.0);
    }
    else if (x < 5.25) {
        // Region 3: (2.7 < x < 5.25) Padé approximant at x = 3.5
        return
        ((((((((0.0001557003672851874 * x + 0.0004957076567581154) * x
              - 0.01817285595260878) * x + 0.1345918024289261) * x
              - 0.5080845404184391) * x + 1.166038409336196) * x
              - 1.60684176885124) * x + 1.289735464215056) * x - 0.3462165267137038)
        /
        ((((((((0.00002594140010637616 * x + 0.0004760781929342163) * x
              - 0.006324137050777044) * x + 0.0323306563822835) * x
              - 0.07455674581922889) * x + 0.04783923527713085) * x
              + 0.1835336998076187) * x - 0.4337222340341569) * x + 0.4043596458474212);
    }
    else if (x < 25) {
        // Region 4: (5.25 < x < 25) Minimax rational approximation
        return
        ((((((-0.000012646774171226343 * x - 0.0032044920889113934) * x
              - 0.13234406270456517) * x - 1.1625737290176312) * x
              - 0.59362239172580916) * x + 3.9213507361249824) * x - 1.1365522141506118)
        /
        ((((((-0.0000015383811528251003 * x - 0.00055217190487695694) * x
              - 0.030985766166024800) * x - 0.41015487838362448) * x
              - 1.0136411717336077) * x + 1.1638470099616859) * x + 1.0);
    }
    else {
        // Region 5: (x > 25) Asymptotic expansion
        double x2 = x * x;
        double x4 = x2 * x2;
        double x6 = x4 * x2;

        return
        (-5 - 3 * x2 - 4 * x4 +
          2 * exp(-x2) * sqrt(M_PI * x2) * (3 - 2 * x2 + 4 * x4) +
          8 * x6 * (0.5772156649015328606 + log(4)))
        / (16. * x6)
        + log(x);
    }
}

void generate_table(const std::string& filename, double x_min, double x_max, int num_points) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  file << std::setprecision(18) << std::scientific; // High precision output
  file << "x,approximation\n";  // CSV Header

  for (int i = 0; i < num_points; ++i) {
    double x = x_min * std::pow(x_max / x_min, static_cast<double>(i) / (num_points - 1));
    double approx_x = hypergeo_approx(x);
    file << x << "," << approx_x << "\n";
  }

  file.close();
  std::cout << "Table generated: " << filename << std::endl;
}

void generate_table_single(const std::string& filename, double x_min, double x_max, int num_points) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  file << std::setprecision(8) << std::scientific; // High precision output
  file << "x,approximation\n";  // CSV Header

  for (int i = 0; i < num_points; ++i) {
    double x = x_min * std::pow(x_max / x_min, static_cast<double>(i) / (num_points - 1));
    double approx_x = hypergeo_approx_single(x);
    file << x << "," << approx_x << "\n";
  }

  file.close();
  std::cout << "Table generated: " << filename << std::endl;
}

void time_approximation(const double x_min, const double x_max, const int num_points) {
  const double dx = (x_max - x_min) / double(num_points - 1);
  auto start = std::chrono::high_resolution_clock::now();
  double sum_1 = 0.;
  for (int i = 0; i < num_points; ++i)
    sum_1 += hypergeo_approx(x_min + static_cast<double>(i) * dx);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time for hypergeo_approx: " << elapsed.count() << " seconds  (sum = " << sum_1 << ")" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  double sum_2 = 0.;
  for (int i = 0; i < num_points; ++i)
    sum_2 += hypergeo_approx_single(x_min + static_cast<double>(i) * dx);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Time for hypergeo_approx_single: " << elapsed.count() << " seconds  (sum = "
            << sum_2 << " = " << 100.*(1. - static_cast<double>(sum_2) / sum_1) << " % error)" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  double sum_3 = 0.;
  for (int i = 0; i < num_points; ++i)
    sum_3 += old_hypergeo_approx(x_min + static_cast<double>(i) * dx);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Time for old_hypergeo_approx: " << elapsed.count() << " seconds  (sum = "
            << sum_3 / sqrt(M_PI) << " = " << 100.*(1. - sum_3 / sqrt(M_PI) / sum_1) << " % error)" << std::endl;
}

int main(int argc, char** argv) {
  // The original benchmark used 500 million samples. Require an explicit
  // invocation so accidentally starting this utility is safe on a login node.
  if (argc != 3 || std::string(argv[1]) != "--benchmark") {
    std::cout << "Usage: hypergeo_approx --benchmark NUMBER_OF_SAMPLES\n";
    return 0;
  }
  const int samples = std::stoi(argv[2]);
  if (samples < 2) {
    std::cerr << "NUMBER_OF_SAMPLES must be at least 2\n";
    return 1;
  }
  time_approximation(25., 100., samples);
  return 0;
}
