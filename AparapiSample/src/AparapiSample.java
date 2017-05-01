import java.util.Random;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;

public class AparapiSample {
	public static void main(String[] args) {
		final int MAT_SIZE = 2000;
		float[] a = new float[MAT_SIZE * MAT_SIZE];
		float[] b = new float[MAT_SIZE * MAT_SIZE];

		Random rand = new Random(0);
		for(int i = 0; i < a.length; i++){
			a[i] = rand.nextFloat() * 100;
			b[i] = rand.nextFloat() * 100;
		}

		MatMulKernel mmk = new MatMulKernel(a, b, MAT_SIZE);
		float[] cCpu = new float[MAT_SIZE * MAT_SIZE];

		System.out.println("Begin gpu.");
		long timeGpu = System.nanoTime();

		// calc by gpu
		float[] cGpu = mmk.calculate();

		timeGpu = System.nanoTime() - timeGpu;

		System.out.println("Begin cpu.");
		long timeCpu = System.nanoTime();

		// calc by cpu
		matmul(a, b, cCpu, MAT_SIZE);

		timeCpu = System.nanoTime() - timeCpu;

		System.out.println("Fin calc.");

		float diff = 0;
		for(int i = 0; i < cCpu.length; i++){
			diff += sq(cCpu[i] - cGpu[i]);
		}

//		dump(cGpu, MAT_SIZE);
//		dump(cCpu, MAT_SIZE);

		System.out.println("Diff/SIZE: " + (diff / sq(MAT_SIZE)));
		System.out.println("Time GPU: " + timeGpu + "ns");
		System.out.println("Time CPU: " + timeCpu + "ns");
	}

	static void dump(float[] c, int n){
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				System.out.printf("%5.5f ", c[i * n + j]);
			}System.out.println();
		}System.out.println();
	}

	static float sq(float a){
		return a * a;
	}

	static void matmul(float[] a, float[] b, float[] c, int n){
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				float sum = 0;
				for(int k = 0; k < n; k++){
					sum += a[i * n + k] * b[k * n + j];
				}
				c[i * n + j] = sum;
			}
		}
	}
}

class MatMulKernel extends Kernel{
	private float[] a, b, c;
	private int n;
	public MatMulKernel(float[] a, float[] b, int n) {
		this.a = a;
		this.b = b;
		c = new float[n * n];
		this.n = n;
	}

	public float[] calculate(){
		Range range = Range.create(a.length);
		execute(range);
		return c;
	}

	@Override
	public void run() {
		final int id = getGlobalId();
		final int cx = id % n;
		final int cy = id / n;

		float sum = 0;
		for(int i = 0; i < n; i++){
			sum += a[cy * n + i] * b[i * n + cx];
		}
		c[id] = sum;
	}
}