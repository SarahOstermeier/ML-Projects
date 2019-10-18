import java.util.*;

import java.io.*;

/**
 * Implements a small (2 unit hidden layer + 1 unit output) neural net from
 * scratch
 * 
 * @author Sarah Ostermeier
 *
 */
public class Mini_NN {

	// Comma for parsing csv file
	private static final String COMMA = ",";

	/**
	 * Inner class to hold an individual data sample, composed of x, a double array
	 * of features and int y, the associated label (0, or 1) *
	 */
	private static class Sample {
		Double[] x; // an array of features
		Integer y; // an integer label (0 or 1)

		/**
		 * @param x an array of features
		 * @param y an integer label (0 or 1)
		 */
		Sample(Double[] x, Integer y) {
			this.x = x;
			this.y = y;
		}
	}

	/**
	 * Parses input csv file of processed data and returns an ArrayList of type
	 * Sample of the data
	 * 
	 * @param filename
	 * @return ArrayList of type Sample of the data
	 */
	private static ArrayList<Sample> readData(String filename) {

		ArrayList<Sample> data = new ArrayList<>();
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new FileReader(filename));
			String line = "";

			while ((line = reader.readLine()) != null) {
				String[] sample = line.split(COMMA);
				// 1 is added as the first element of the feature array as a bias term
				// 4 first four elements of sample array are added as features
				Double[] x = { 1.0, Double.parseDouble(sample[0]), Double.parseDouble(sample[1]),
						Double.parseDouble(sample[2]), Double.parseDouble(sample[3]) };
				// Final element of sample array becomes the label
				Integer y = Integer.valueOf(sample[4]);

				data.add(new Sample(x, y));
			}
		} catch (Exception e) {
			System.out.println("Error reading file");
			e.printStackTrace();
		} finally {
			try {
				reader.close();
			} catch (IOException e2) {
				System.out.println("Error closing BufferedReader");
				e2.printStackTrace();
			}
		}
		return data;
	}

	/**
	 * @param x
	 * @param w
	 * @return
	 */
	private static double calcZ(Double[] x, Double[] w) {
		double sum = 0;

		for (int i = 0; i < x.length; i++) {
			sum += w[i] * x[i];
		}

		return sum;
	}

	private static double calcA(Double[] x, Double[] w) {
		double z = calcZ(w, x);
		double result = 1 / (1 + Math.exp(-z));
		return result;
	}

	private static double calcNetworkA(Double[] x, Double[] w2_1, Double[] w2_2, Double[] w3_1) {
		Double[] l2 = { 1.0, calcA(x, w2_1), calcA(x, w2_2) };
		return calcA(l2, w3_1);
	}

	private static Double[] pDerivative(Double[] x, Double[] w2_1, Double[] w2_2, Double[] w3_1, int y) {
		Double[] deltas = new Double[3];
		double a2_1 = calcA(x, w2_1);
		double a2_2 = calcA(x, w2_2);
		double a3_1 = calcNetworkA(x, w2_1, w2_2, w3_1);
		double delta3 = (a3_1 - y) * a3_1 * (1 - a3_1);

		deltas[0] = delta3 * w3_1[1] * a2_1 * (1 - a2_1);
		deltas[1] = delta3 * w3_1[2] * a2_2 * (1 - a2_2);
		deltas[2] = delta3;
		return deltas;
	}

	private static ArrayList<Double[]> pdWeights(Double[] x, Double[] w2_1, Double[] w2_2, Double[] w3_1, int y) {
		ArrayList<Double[]> weights = new ArrayList<>();
		Double[] weights2_1 = new Double[5];
		Double[] weights2_2 = new Double[5];
		Double[] weights3_1 = new Double[3];

		Double[] delta = pDerivative(x, w2_1, w2_2, w3_1, y);

		for (int i = 0; i < x.length; i++) {
			weights2_1[i] = delta[0] * x[i];
			weights2_2[i] = delta[1] * x[i];
		}

		weights3_1[0] = delta[2] * 1;
		weights3_1[1] = delta[2] * calcA(x, w2_1);
		weights3_1[2] = delta[2] * calcA(x, w2_2);

		weights.add(0, weights2_1);
		weights.add(1, weights2_2);
		weights.add(2, weights3_1);

		return weights;
	}

	private static ArrayList<Double[]> updateWeights(Double[] x, Double[] w2_1, Double[] w2_2, Double[] w3_1, int y,
			double eta) {
		Double[] update2_1 = new Double[5];
		Double[] update2_2 = new Double[5];
		Double[] update3_1 = new Double[3];

		ArrayList<Double[]> weights = new ArrayList<>();
		ArrayList<Double[]> pd = pdWeights(x, w2_1, w2_2, w3_1, y);

		for (int i = 0; i < pd.get(0).length; i++) {
			update2_1[i] = w2_1[i] - (eta * pd.get(0)[i]);
		}

		for (int i = 0; i < pd.get(1).length; i++) {
			update2_2[i] = w2_2[i] - (eta * pd.get(1)[i]);
		}

		for (int i = 0; i < pd.get(2).length; i++) {
			update3_1[i] = w3_1[i] - (eta * pd.get(2)[i]);
		}

		weights.add(0, update2_1);
		weights.add(1, update2_2);
		weights.add(2, update3_1);

		return weights;
	}

	private static Double[] toArray(ArrayList<Double[]> values) {
		Double[] array = new Double[13];

		for (int i = 0; i < values.get(0).length; i++) {
			array[i] = values.get(0)[i];
		}
		for (int i = 0; i < values.get(1).length; i++) {
			array[i + values.get(0).length] = values.get(1)[i];
		}

		for (int i = 0; i < values.get(2).length; i++) {
			array[i + values.get(0).length + values.get(1).length] = values.get(2)[i];
		}
		return array;
	}

	private static double eval(ArrayList<Sample> eval, Double[] w2_1, Double[] w2_2, Double[] w3_1) {
		double sum = 0;
		for (Sample item : eval) {
			double a3_1 = calcNetworkA(item.x, w2_1, w2_2, w3_1);
			sum += Math.pow((a3_1 - item.y), 2) / 2;
		}
		return sum;
	}

	private static ArrayList<Double[]> SGD(int flag, ArrayList<Sample> train, ArrayList<Sample> eval, Double[] w2_1,
			Double[] w2_2, Double[] w3_1, double eta) {
		ArrayList<Double[]> trained_weights = new ArrayList<>();
		Double[] prev_w2_1 = w2_1;
		Double[] prev_w2_2 = w2_2;
		Double[] prev_w3_1 = w3_1;

		for (Sample item : train) {
			Double[] x = item.x;
			Integer y = item.y;
			ArrayList<Double[]> update = updateWeights(x, prev_w2_1, prev_w2_2, prev_w3_1, y, eta);
			Double error = eval(eval, update.get(0), update.get(1), update.get(2));

			if (flag == 500)
				printSGD(update, error);

			prev_w2_1 = update.get(0);
			prev_w2_2 = update.get(1);
			prev_w3_1 = update.get(2);
		}

		trained_weights.add(0, prev_w2_1);
		trained_weights.add(1, prev_w2_2);
		trained_weights.add(2, prev_w3_1);

		return trained_weights;
	}

	private static void printSGD(ArrayList<Double[]> weight_vals, double eval) {
		Double[] w = toArray(weight_vals);

		System.out.printf("%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n", w[0], w[1], w[2], w[3],
				w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12]);

		System.out.printf("%.5f\n", eval);
	}

	public static void main(String[] args) {

		int flag = Integer.valueOf(args[0]);

		String train_file = "./train.csv";
		String test_file = "./test.csv";
		String eval_file = "./eval.csv";

		ArrayList<Sample> train = readData(train_file);
		ArrayList<Sample> test = readData(test_file);
		ArrayList<Sample> eval = readData(eval_file);

		Double[] w2_1 = { Double.valueOf(args[1]), Double.valueOf(args[2]), Double.valueOf(args[3]),
				Double.valueOf(args[4]), Double.valueOf(args[5]) };
		Double[] w2_2 = { Double.valueOf(args[6]), Double.valueOf(args[7]), Double.valueOf(args[8]),
				Double.valueOf(args[9]), Double.valueOf(args[10]) };
		Double[] w3_1 = { Double.valueOf(args[11]), Double.valueOf(args[12]), Double.valueOf(args[13]) };

		if (flag == 100) {
			Double[] x = { 1.0, Double.valueOf(args[14]), Double.valueOf(args[15]), Double.valueOf(args[16]),
					Double.valueOf(args[17]) };

			System.out.printf("%.5f %.5f\n", calcA(x, w2_1), calcA(x, w2_2));
			System.out.printf("%.5f\n", calcNetworkA(x, w2_1, w2_2, w3_1));
		}

		if (flag == 200) {
			Double[] x = { 1.0, Double.valueOf(args[14]), Double.valueOf(args[15]), Double.valueOf(args[16]),
					Double.valueOf(args[17]) };
			int y = Integer.valueOf(args[18]);
			System.out.printf("%.5f\n", pDerivative(x, w2_1, w2_2, w3_1, y)[2]);
		}

		if (flag == 300) {
			Double[] x = { 1.0, Double.valueOf(args[14]), Double.valueOf(args[15]), Double.valueOf(args[16]),
					Double.valueOf(args[17]) };
			int y = Integer.valueOf(args[18]);
			Double[] delta_vals = pDerivative(x, w2_1, w2_2, w3_1, y);
			System.out.printf("%.5f %.5f\n", delta_vals[0], delta_vals[1]);
		}

		if (flag == 400) {
			Double[] x = { 1.0, Double.valueOf(args[14]), Double.valueOf(args[15]), Double.valueOf(args[16]),
					Double.valueOf(args[17]) };
			int y = Integer.valueOf(args[18]);
			Double[] weights3_1 = pdWeights(x, w2_1, w2_2, w3_1, y).get(2);
			Double[] weights2_1 = pdWeights(x, w2_1, w2_2, w3_1, y).get(0);
			Double[] weights2_2 = pdWeights(x, w2_1, w2_2, w3_1, y).get(1);

			System.out.printf("%.5f %.5f %.5f\n", weights3_1[0], weights3_1[1], weights3_1[2]);
			System.out.printf("%.5f %.5f %.5f %.5f %.5f\n", weights2_1[0], weights2_1[1], weights2_1[2], weights2_1[3],
					weights2_1[4]);
			System.out.printf("%.5f %.5f %.5f %.5f %.5f\n", weights2_2[0], weights2_2[1], weights2_2[2], weights2_2[3],
					weights2_2[4]);
		}

		if (flag == 500) {
			double eta = Double.valueOf(args[14]);

			SGD(flag, train, eval, w2_1, w2_2, w3_1, eta);
		}

		if (flag == 600) {
			double eta = Double.valueOf(args[14]);
			ArrayList<Double[]> weights = SGD(flag, train, eval, w2_1, w2_2, w3_1, eta);
			int correct = 0;

			for (Sample item : test) {
				Double[] x = item.x;
				Integer y = item.y;
				Integer pred_y;
				double a = calcNetworkA(x, weights.get(0), weights.get(1), weights.get(2));
				if (a <= 0.5)
					pred_y = 0;
				else
					pred_y = 1;
				if (pred_y == y)
					correct++;

				System.out.print(y + " ");
				System.out.print(pred_y + " ");
				System.out.printf("%.5f\n", a);
			}
			System.out.printf("%.2f\n", (double) correct / test.size());
		}
	}
}
