/***********************************************************************/
/*                                                                     */
/*   svm_struct_classify.c                                             */
/*                                                                     */
/*   Classification module of SVM-struct.                              */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*   Modified by: Yu Xiang                                             */
/*   Date: 05.01.12                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/************************************************************************/

#include <stdio.h>
#include <mpi.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "svm_common.h"
#ifdef __cplusplus
}
#endif
#include "svm_struct_api.h"
#include "select_gpu.h"

char testfile[200];
char cadfile[200];
char modelfile[200];
char resultfile[200];
char predictionsfile[200];
char predictionsfile_local[200];

void read_input_parameters(int, char **, char *, char *, char *, long *, STRUCT_LEARN_PARM *);
void print_help(void);

int main (int argc, char* argv[])
{
  int correct = 0, incorrect = 0, no_accuracy = 0;
  int i, j;
  int num, count, cad_num;
  double t1, runtime = 0;
  double avgloss = 0, l;
  double result[3], result_sum[3];
  float azimuth_diff;
  CAD **cads;
  FILE *predfl;
  STRUCTMODEL model; 
  STRUCT_LEARN_PARM sparm;
  STRUCT_TEST_STATS teststats;
  SAMPLE testsample;
  LABEL *y = NULL;

  /* MPI process */
  int rank, n;
  int procs_num;
  int start, end, block_size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs_num);

  /* select GPU */
  select_gpu(rank);

  svm_struct_classify_api_init(argc, argv);

  read_input_parameters(argc, argv, testfile, modelfile, predictionsfile, &verbosity, &sparm);

  /* read cad models */
  cads = read_cad_model(cadfile, &cad_num, 0, &sparm);

  printf("Test the final model\n");

  printf("Read model\n");
  model = read_struct_model(modelfile, &sparm);
  printf("Read model done\n");
  /* set the cad model for structmodel */
  model.cad_num = cad_num;
  model.cads = cads;

  printf("Read test examples\n");
  testsample = read_struct_examples_batch(testfile, &sparm, &model);
  printf("Read test examples done\n");

  printf("Classifying test examples..");

  n = testsample.n;
  block_size = (n+procs_num-1) / procs_num;
  start = rank*block_size;
  end = start+block_size-1 > n-1 ? n-1 : start+block_size-1;

  /* open file to write outpyt */
  sprintf(predictionsfile_local, "%s_%d", predictionsfile, rank);
  printf("Writing predictions to file %s..\n", predictionsfile_local);
  if ((predfl = fopen (predictionsfile_local, "w")) == NULL)
  { 
    perror(predictionsfile_local); 
    exit (1); 
  }

  correct = 0;
  incorrect = 0;
  no_accuracy = 0;
  avgloss = 0;
  for(i = start; i <= end; i++) 
  {
    t1=get_runtime();
    printf("Test example %d\n", i+1);

    y = classify_struct_example(testsample.examples[i].x, &num, 1, &model, &sparm);
/*
    y = (LABEL*)my_malloc(sizeof(LABEL));
    num = 1;
    y[0] = find_most_positive_constraint(testsample.examples[i].x, testsample.examples[i].y, &model, &sparm);
*/
    runtime += (get_runtime()-t1);

    count = 0;
    for(j = 0; j < num; j++)
    {
      if(y[j].object_label)
        count++;
    }

    /* write predictions */
    fprintf(predfl, "%d\n", count);
    if(count == 0)
      continue;
    for(j = 0; j < num; j++)
    {
      if(y[j].object_label)
        write_label(predfl, y[j], &model, &sparm);
    }

    l = loss(testsample.examples[i].y, y[0], &sparm);
    avgloss += l;
    azimuth_diff = compute_azimuth_difference(testsample.examples[i].y, y[0], &model);

    if(testsample.examples[i].y.object_label == y[0].object_label && (azimuth_diff < 22.5 || 360-azimuth_diff < 22.5)) 
      correct++;
    else
      incorrect++;
    eval_prediction(i, testsample.examples[i], y[0], &model, &sparm, &teststats);

    if(empty_label(testsample.examples[i].y)) 
    { 
      no_accuracy = 1;
    } /* test data is not labeled */
    if(verbosity >= 2)
    {
      if((i+1) % 100 == 0)
      {
        printf("%d..",i+1);
        fflush(stdout);
      }
    }

    for(j = 0; j < num; j++)
      free_label(y[j]);
    free(y);
  }  /* end forloop for test examples */

  result[0] = correct;
  result[1] = incorrect;
  result[2] = avgloss;
  MPI_Allreduce(result, result_sum, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  correct = result_sum[0];
  incorrect = result_sum[1];
  avgloss = result_sum[2];

  avgloss /= testsample.n;
  fclose(predfl);
  MPI_Barrier(MPI_COMM_WORLD);

  if(verbosity >= 2)
  {
    printf("done\n");
    printf("Runtime (without IO) in cpu-seconds: %.2f\n", (float)(runtime/100.0));    
  }
  if((rank == 0 && !no_accuracy) && (verbosity >= 1))
  {
    printf("Average loss on test set: %.4f\n",(float)avgloss);
    printf("Zero/one-error on test set: %.2f%% (%d correct, %d incorrect, %d total)\n", (float)100.0*incorrect/testsample.n, correct, incorrect, testsample.n);
  }
  print_struct_testing_stats(testsample, &model, &sparm, &teststats);

  if(rank == 0)
    combine_files(resultfile, predictionsfile, procs_num);
  MPI_Barrier(MPI_COMM_WORLD);

  free_struct_sample(testsample);
  free_struct_model(model);

  for(i = 0; i < cad_num; i++)
    destroy_cad(cads[i]);
  free(cads);

  svm_struct_classify_api_exit();

  MPI_Finalize();

  return(0);
}

void read_input_parameters(int argc, char **argv, char *testfile, 
			   char *modelfile, char *predictionsfile, 
			   long int *verbosity, STRUCT_LEARN_PARM *struct_parm)
{
  int i, len;
  
  /* set default */
  strcpy(modelfile, "data/svm_model");
  strcpy(predictionsfile, "tmp/svm_predictions"); 
  (*verbosity) = 2;

  for(i = 1; (i<argc) && ((argv[i])[0] == '-'); i++) 
  {
    switch ((argv[i])[1]) 
    { 
      case 'h': print_help(); exit(0);
      case 'v': i++; (*verbosity) = atol(argv[i]); break;
      case '-': parse_struct_parameters_classify(argv[i],argv[i+1]);i++; break;
      default: printf("\nUnrecognized option %s!\n\n",argv[i]);
	       print_help();
	       exit(0);
    }
  }
  if((i+1) >= argc)
  {
    printf("\nNot enough input parameters!\n\n");
    print_help();
    exit(0);
  }
  strcpy(testfile, argv[i]);
  i++;
  strcpy(cadfile, argv[i]);
  i++;
  strcpy(modelfile, argv[i]);

  /* construct class name */
  len = strlen(testfile);
  strncpy(struct_parm->cls, testfile, len-4);
  struct_parm->cls[len-4] = '\0';

  if(i+1 < argc)
  {
    i++;
    strcpy(resultfile, argv[i]);
  }
  else
    sprintf(resultfile, "%s.pre", struct_parm->cls);

  struct_parm->cset.lhs = NULL;
  struct_parm->cset.rhs = NULL;
  struct_parm->cset.m = 0;
}

void print_help(void)
{
  printf("\nSVM-struct classification module: %s, %s, %s\n",INST_NAME,INST_VERSION,INST_VERSION_DATE);
  printf("   includes SVM-struct %s for learning complex outputs, %s\n",STRUCT_VERSION,STRUCT_VERSION_DATE);
  printf("   includes SVM-light %s quadratic optimizer, %s\n",VERSION,VERSION_DATE);
  copyright_notice();
  printf("   usage: svm_struct_classify [options] example_file model_file output_file\n\n");
  printf("options: -h         -> this help\n");
  printf("         -v [0..3]  -> verbosity level (default 2)\n\n");

  print_struct_help_classify();
}
