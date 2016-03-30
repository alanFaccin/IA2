//============================================================================
// Name        : Percepton.cpp
// Author      : Alan Jhones
// Version     :
// Copyright   : Your copyright notice
// Description :Estruturação de Rede Neural MLP
//============================================================================

#include <stdio.h>
//#include <conio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include  <time.h>

int padroes = 1, funcao = 1, contador = 0, epocas = 0, fim = 0;
double net = 0, y = 0, TaxaAprendizado = 0.005, erro_medio_quadratico = 0;

// Padrıes de entrada da rede (sensores).
float padrao[11][2] = { 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0 };

// Valores desejados dos padrıes ao final do treinamento.
float d[11][1] = { 0.0, 1.0, 1.0, 0.0 };

const int input = 4; // quantidade de entardas para a camada 1
const int camada_1 = 2; // quantidade  de nuronios na camada 1(intermediaria)
const int camada_2 = 1; // quantidade de neuronios na camada 2 (saida)

double PesosCamada_1[input * camada_1];
double PesosCamada_2[camada_1 * camada_2];

double DeltaPesosCamada_1[input * camada_1];
double DeltaPesosCamada_2[camada_1 * camada_2];

double entrada_camada1[camada_1] = { 0 }, saida_camada1[camada_1] = { 0 },
		erro_camada1[camada_1] = { 0 };
double entrada_camada2[camada_2] = { 0 }, saida_camada2[camada_2] = { 0 },
		erro_camada2[camada_2] = { 0 };
double saidas[camada_2] = { 0 };

double funcaoDeAtivacao(double net, int funcao, double a) {
	double valor = 0;

	if (!funcao) {
		/*
		 1
		 y(n) = ---------------
		 1 + exp(-a.net)
		 */

		valor = (1.0 / (1.0 + exp(-a * net)));

		return (valor);
	} else { // funcao hiperbólica
		/*
		 exp(a.net) - exp(-a.net)
		 y(n) = ------------------------
		 exp(a.net) + exp(-a.net)
		 */

		valor =
				((exp(a * net) - exp(-a * net)) / (exp(a * net) + exp(-a * net)));

		return (valor);
	}

	return (valor);
}

float derivada(float net, int funcao, float a) {

	if (!funcao) {

		/*
		 1                       1
		 y(n) = --------------- * ( 1 - --------------- )
		 1 - exp(-a.net)         1 - exp(-a.net)
		 */

		return ((1.0 / (1.0 + exp(-a * net)))
				* (1.0 - (1.0 / (1.0 + exp(-a * net)))));

	} else {

		/*
		 exp(a.net) - exp(-a.net)
		 y(n) = 1 - ( ------------------------ )≤
		 exp(a.net) + exp(-a.net)

		 */

		return (1.0
				- pow(
						(exp(a * net) - exp(-a * net))
								/ (exp(a * net) + exp(-a * net)), 2));
	}
}

double CalculoCamada(int camada, int iteracao) {
	double somatorio = 0;
	int n = 0;
	if (camada == 1) {
		for (int j = 0; j < camada_1; j++) {
			somatorio = 0;
			for (int i = 0; i < input; i++) {
				somatorio += DeltaPesosCamada_1[n] * padrao[iteracao][i];
				n += 1;
			}
			entrada_camada1[j] = somatorio;
			saida_camada1[j] = funcaoDeAtivacao(entrada_camada1[j], funcao,
					1.0);
		}

	} else if (camada == 2) {
		n = 0;
		for (int j = 0; j < camada_2; j++) {
			somatorio = 0;
			for (int i = 0; i < camada_1; i++) {
				somatorio += DeltaPesosCamada_2[n] * saida_camada1[i];
				n += 1;
			}
			entrada_camada2[j] = somatorio;
			saida_camada2[j] = funcaoDeAtivacao(entrada_camada2[j], funcao,
					1.0);
		}

	}
	return somatorio;
}

void update_Pesos(int camada, int iteracao) {
	int n;
	if (camada == 1) {

		for (int i = 0; i < input; i++) {
			n = 0;
			for (int j = 0; j < camada_1; j++) {
				DeltaPesosCamada_1[n + i] = TaxaAprendizado
						* padrao[iteracao][i] * erro_camada1[j];
				PesosCamada_1[n + i] = PesosCamada_1[n + i]
						+ DeltaPesosCamada_1[n + i];
				n += input;
			}
		}

	} else if (camada == 2) {

		for (int i = 0; i < camada_1; i++) {
			n = 0;
			for (int j = 0; j < camada_2; j++) {
				DeltaPesosCamada_2[n + i] = TaxaAprendizado * saida_camada1[i]* erro_camada2[j];
				PesosCamada_2[n + i] = PesosCamada_2[n + i]
						+ DeltaPesosCamada_2[n + i];
				;
				n += camada_1;
			}
		}
	}
}
/*função reponsável por realizar o cálculo do erro quadrático, levando em conta a matriz de valor desejado e a saida da camada 2*/
double calc_erro_quadratico(int iteracao) {

	double err = 0;
	int j = 0;
	for (j = 0; j < camada_2; j++) {
		err += pow((d[iteracao][j] - saida_camada2[j]), 2);
	}
	return err;
}
double calc_erro_medio_quadratico(double erro_qud, double erro_m_quad) {
	erro_m_quad += (0.5 * erro_qud);
	return erro_m_quad;
}
/*Funçao responsavel por zerar o array de pesos da camada desejada*/
void zeraPesos(int camada) {
	int j = 0;
	if (camada == 1) {
		for (j = 0; j < (input * camada_1); j++) {
			PesosCamada_1[j] = 0.0;
		}

	} else if (camada == 2) {
		for (j = 0; j < (camada_1 * camada_2); j++) {
			PesosCamada_2[j] = 0.0;
		}
	}

}

void zeraVetores(int camada) {
	int j = 0;
	if (camada == 1) {
		for (j = 0; j < camada_1; j++) {
			entrada_camada1[j] = 0.0;
			saida_camada1[j] = 0.0;
			erro_camada1[j] = 0.0;
		}

	} else if (camada == 2) {
		for (j = 0; j < camada_2; j++) {
			entrada_camada2[j] = 0.0;
			saida_camada2[j] = 0.0;
			erro_camada2[j] = 0.0;
		}
	}

}

void randomiza_Pesos(int camada) {
	int j = 0;
	if (camada == 1) {
		for (j = 0; j < (input * camada_1); j++) {
			PesosCamada_1[j] = (float) (rand() % 100) / 1000;
		}

	} else if (camada == 2) {
		for (j = 0; j < (camada_1 * camada_2); j++) {
			PesosCamada_2[j] = (float) (rand() % 100) / 1000;
		}
	}

}

void grava_Pesos_inicials_arquivo(int camada) {

	int j = 0;
	FILE *fp;
	fp = fopen("pesos_rand.txt", "a+");
	if (camada == 1) {
		fprintf(fp, "Pesos Camada 1\n");
		for (j = 0; j < (input * camada_1); j++) {
			fprintf(fp, "%f\n", PesosCamada_1[j]);
		}

	} else if (camada == 2) {
		fprintf(fp, "Pesos Camada 2\n\n");
		for (j = 0; j < (camada_1 * camada_2); j++) {
			fprintf(fp, "%f\n", PesosCamada_2[j]);
		}
	}
	fclose(fp);
}

void calcula_erro(int camada, int iteracao) {

	if (camada == 1) {
		double somatorio = 0;
		int n;
		for (int i = 0; i < camada_1; i++) {
			n = 0;
			somatorio = 0;
			for (int j = 0; j < camada_2; j++) {
				somatorio += (erro_camada2[j] * DeltaPesosCamada_2[n + i]);
				n += camada_1;
			}
			erro_camada1[i] = somatorio
					* derivada(entrada_camada1[i], funcao, 1.0);
		}
	} else if (camada == 2) {
		for (int i = 0; i < camada_2; i++) {
			erro_camada2[i] = (d[iteracao][i] -saida_camada2[i])* derivada(entrada_camada2[i],funcao,1.0);
			//erro_camada2[i] = d[iteracao][i]- *derivada(entrada_camada2[i], funcao, 1.0);
		}
	}
}
void grava_pesos_apos_treinamento() {

	//****************** Gravacao dos Pesos Apos treinamento ******************
	FILE *fp;
	fp = fopen("pesos_treino.txt", "wt");
	// camada 1
	fprintf(fp, "Pesos Camada 1\n");
	for (int j = 0; j < (input * camada_1); j++) {
		fprintf(fp, "%f\n", PesosCamada_1[j]);
	}
	// camada 2
	fprintf(fp, "Pesos Camada 2\n\n");
	for (int j = 0; j < (camada_1 * camada_2); j++) {
		fprintf(fp, "%f\n", PesosCamada_2[j]);
	}
	fclose(fp);
}
void treinamento() {
	FILE *fp;
	fp = fopen("treinamento.txt", "wt");
	epocas 		=	1000;
	padroes 	=	4;
	contador	=	0;
	do {
		contador++;
		//Laço para a propagação dos padroes pela rede.

		for (int k = 0; k < padroes; k++) {
			//**************************** Forward **************************
			CalculoCamada(1, k);	//obter as entardas e saidas da camada 1
			CalculoCamada(2, k); // obter as netradas e saidas da camada 2

			erro_medio_quadratico = calc_erro_medio_quadratico(calc_erro_quadratico(k),
					erro_medio_quadratico);

			//**************************** BackWard **************************

			//Calculo do erro para camada 2.
			calcula_erro(2, k);
			//Atualizacao dos pesos para camada 2.
			update_Pesos(2, k);
			//Calculo do erro para camada 1.
			calcula_erro(1, k);
			//Atualizacao dos pesos para camada 1.
			update_Pesos(1, k);

		}
		// Calculo do erro médio quadratico da época de treinamento.
		erro_medio_quadratico = (1.0 / padroes) * erro_medio_quadratico;

		printf("%d\t%f\n", (int) contador, erro_medio_quadratico);

		fprintf(fp, "%d\t%f\n", (int) contador, erro_medio_quadratico);

		erro_medio_quadratico = 0;

	} while (!fim && contador != epocas);

	// Fecha o ponteiro do arquivo de erros de treinamento.
	fclose(fp);
	grava_pesos_apos_treinamento();
}

int main() {

	/* initialize random seed: */
	srand(time(NULL));

	zeraPesos(1);
	zeraPesos(2);

	zeraVetores(1);
	zeraVetores(2);

	randomiza_Pesos(1);
	randomiza_Pesos(2);

	grava_Pesos_inicials_arquivo(1);
	grava_Pesos_inicials_arquivo(2);

	treinamento();
}
