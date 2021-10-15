/*

Copyright (c) 2020, Douglas Santry
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, is permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <softmaxNNm.h>

#define N_POINTS	30

#define PI 			3.141592653589793
#define PI_2		1.570796326794897
#define PI_DELTA	(PI_2 / N_POINTS)

/*
 * (x, y) in quadrants.
 *
 */

double polarQ [] = {
-2.29240727171952, -1.48424491989863, 3, 
 1.45088565409287, -1.32321551055482, 4, 
 0.808601867799222, -2.32796539215308, 4, 
 0.183481694256994, -2.05766224971618, 4, 
 -0.204291507285715, -1.81937303344634, 3, 
 0.301449522727812, -0.768232024664306, 4, 
 1.23387543994358, 1.09771393780718, 1, 
 -1.84957767478188, -0.0640464765818828, 3, 
 1.344300515513, 0.036052920017901, 1, 
 0.393539945144457, 2.03919458048562, 1, 
 1.81117538829371, -0.988161239363614, 4, 
 -0.654034443931663, -0.0969826508846382, 3, 
 0.797384014510628, 0.743364727616669, 1, 
 -1.00447808294127, 0.551442894725597, 2, 
 -0.310728947072935, 0.793562101162612, 2, 
 1.55619381639037, 2.2686695526798, 1, 
 1.99519335633803, -1.55412889908004, 4, 
 0.17573892567568, 0.507665140740326, 1, 
 -0.471425470836951, -1.00721430045691, 3, 
 -2.01966127968543, -0.622124532304338, 3, 
 -0.440057045305845, -1.88674615688435, 3, 
 -2.57387266147432, 0.958297481622516, 2, 
 -0.554651674369894, 1.23332371281191, 2, 
 -0.709584872227339, 0.401179837712257, 2, 
 1.77271611380485, -1.50125556804123, 4, 
 0.0795724478686849, -0.855147434569206, 4, 
 -1.34890934965629, 1.04308948480022, 2, 
 1.21392363824006, -2.18571415798367, 4, 
 0.355235896576844, -2.71029654718674, 4, 
 0.730601609562618, -0.62686190817954, 4
};

DataSet_t *BuildTrainingSet (void);
void Run (int *);

int main (int argc, char *argv[])
{
	if (argc < 2)
	{
		printf ("Usage: LoW hidden-layers output-layers\n");
		exit (-1);
	}

	long seed = time (0);
	printf ("Seed %ld\n", seed);

	srand (seed);

	int N_layers = argc - 1;
	int *layers = new int [N_layers + 2];	// widths plus length prefix, inputs
	layers[0] = N_layers + 1;
	layers[1] = 2;							// 2 input
	for (int i = 0; i < N_layers; ++i)
		layers[i + 2] = atoi (argv[i + 1]);

	Run (layers);

	delete [] layers;
}

void Run (int *layers)
{
	DataSet_t *O = BuildTrainingSet ();
	SoftmaxNNm_t *Np = NULL;
	double guess;

	Np = new SoftmaxNNm_t (layers + 1, layers[0], RPROP);

	Np->SetHalt (1e-2);

	try {

		Np->Train (O, 10000);

	} catch (const char *excep) {

		printf ("ERROR: %s\n", excep);
	}

	printf (" *** Loss\t%f\t%f\t%d\n",
		Np->Loss (),
		Np->Accuracy (),
		Np->Steps ());

	bool accept_soln = true;
	bool correct;

	printf ("\t\tTrain\tGuess\tCorrect\n");

	for (int i = 0; i < N_POINTS; ++i)
	{
		guess = Np->Compute ((*O)[i]);

		correct = (*O)[i][2] == guess;
		if (!correct)
			accept_soln = false;

		printf ("DJS_RESULT\t%d\t%d\t%c\n",
			(int) (*O)[i][2],
			(int) guess,
			(correct ? ' ' : 'X'));
	}

	if (accept_soln)
		printf (" *** Solution ACCEPTED.\n");
	else
		printf (" *** Solution REJECTED.\n");

}

DataSet_t *BuildTrainingSet (void)
{
	DataSet_t *O = new DataSet_t (N_POINTS, 2, 1);

	for (int i = 0, index = 0; i < N_POINTS; ++i, index += 3)
	{
		(*O)[i][0] = polarQ [index];
		(*O)[i][1] = polarQ [index + 1];
		(*O)[i][2] = polarQ [index + 2] - 1; // classes must start at zero
	}

	return O;
}

