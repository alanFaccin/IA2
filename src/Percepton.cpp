//============================================================================
// Name        : Percepton.cpp
// Author      : Alan Jhones
// Version     :
// Copyright   : Your copyright notice
// Description :
//============================================================================

#include <stdio.h>
//#include <conio.h>
#include <string.h>
#include <math.h>

int padroes = 1, funcao = 0, contador = 0, epocas = 0, fim = 0;
double net = 0, y = 0, erro = 0, ValorDesejado = 0, TaxaAprendizado = 0.005,
		Entradas[4], DeltaPesos[4], Pesos[4];




// Padrıes de entrada da rede (sensores).
float p[11][2] =
{
0.0,    0.0,
0.0,    1.0,
1.0,    0.0,
1.0,    1.0
};

// Valores desejados dos padrıes ao final do treinamento.
float d[11][1] =
{
0.0,
1.0,
1.0,
0.0
};

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
	} else {
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

double calcula_net(int qtd_entradas, double *input, double *pesos) {
	//Calculo do net da saida do neuronio
	double somatorio = 0;
	for (int a = 0; a < qtd_entradas; a++) {
		somatorio += input[a] * pesos[a];
	}
	return somatorio;
}

void update_Pesos(int qtd_entradas) {

	for (int a = 0; a < qtd_entradas; a++) {
		DeltaPesos[a] = TaxaAprendizado * erro * Entradas[a];
		Pesos[a] = Pesos[a] + DeltaPesos[a];
	}
}
/*função reponsável por realizar o cálculo do erro quadrático, levando em conta a matriz de valor desejado e a saida da camada 2*/
double calc_erro_quadratico(int k) {

	double err = 0;
	int j=0;
	for (j = 0; j < camada_2; j++) {
		err += pow((d[k][j] - saida_camada2[j]), 2);
	}
	return err;
}
/*Funçao responsavel por zerar o array de pesos da camada desejada*/
void zeraPesos(int camada) {
	int j = 0;
	if (camada = 1) {
		for (j = 0; j < (input * camada_1); j++) {
			PesosCamada_1[j] = 0.0;
		}

	} else if (camada = 2) {
		for (j = 0; j < (camada_1 * camada_2); j++) {
			PesosCamada_2[j] = 0.0;
		}
	}

}

void zeraVetores(int camada) {
	int j = 0;
	if (camada = 1) {
		for (j = 0; j < camada_1; j++) {
			entrada_camada1[j] = 0.0;
			saida_camada1[j] = 0.0;
			erro_camada1[j] = 0.0;
		}

	} else if (camada = 2) {
		for (j = 0; j < camada_2; j++) {
			entrada_camada2[j] = 0.0;
			saida_camada2[j] = 0.0;
			erro_camada2[j] = 0.0;
		}
	}

}

void randomiza_Pesos(int camada) {
	int j = 0;
	if (camada = 1) {
		for (j = 0; j < (input * camada_1); j++) {
			PesosCamada_1[j] = (float) (rand() % 100) / 1000;
		}

	} else if (camada = 2) {
		for (j = 0; j < (camada_1 * camada_2); j++) {
			PesosCamada_2[j] = (float) (rand() % 100) / 1000;
		}
	}

}

void grava_Pesos_inicials_arquivo(int camada) {

	int j = 0;
	FILE *fp;
	fp = fopen("pesos_rand.txt", "wt");
	if (camada = 1) {
		fprintf(fp, "Pesos Camada 1\n");
		for (j = 0; j < (input * camada_1); j++) {
			fprintf(fp, "%f\n", PesosCamada_1[j]);
		}

	} else if (camada = 2) {
		fprintf(fp, "Pesos Camada 2\n\n");
		for (j = 0; j < (camada_1 * camada_2); j++) {
			fprintf(fp, "%f\n", PesosCamada_2[j]);
		}
	}
	fclose(fp);
}

void grava_erros_treinamento() {
	FILE *fp;
	fp = fopen("treinamento.txt", "wt");
}
void treinamento() {
	do {
		contador++;

	} while (!fim && contador != epocas);
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
}
