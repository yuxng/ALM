/***********************************************************************/
/*                                                                     */
/*   svm_struct_main.c                                                 */
/*                                                                     */
/*   Command line interface to the alignment learning module of the    */
/*   Support Vector Machine.                                           */
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
/***********************************************************************/


/* the following enables you to use svm-learn out of C++ */
#ifdef __cplusplus
extern "C" {
#endif
#include "svm_common.h"
#include "svm_learn.h"
#ifdef __cplusplus
}
#endif
# include "svm_struct_learn.h"
# include "svm_struct_common.h"
# include "svm_struct_api.h"

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include "select_gpu.h"

void random_negative_samples(char *filename, char *trainfile_wrap, char *trainfile_unwrap, char *trainfile_negative, char *modelfile, int is_wrap, float *overlaps, int cad_num, CAD **cads);
void data_mining_hard_examples(char *filename, char *trainfile, char *testfile, char *modelfile, SAMPLE testsample, int cad_num, CAD **cads);
int compare_energy(const void *a, const void *b);
void read_input_parameters(int, char **, char *, char *, char *, char *, char *, long *, long *, STRUCT_LEARN_PARM *, LEARN_PARM *, KERNEL_PARM *, int *);
void wait_any_key(void);
void print_help(void);
int is_file_exist(char *filename);
int is_confile_same(char* confile_read, char* confile_write);

int main (int argc, char* argv[])
{
  FILE *fp;
  int i, cad_num, iter, flag, num;
  char filename[256], filename_data[256];
  CAD **cads, *cad;
  SAMPLE sample, sample_negative;  /* training sample */
  LEARN_PARM learn_parm;
  KERNEL_PARM kernel_parm;
  STRUCT_LEARN_PARM struct_parm;
  STRUCTMODEL structmodel;
  int alg_type;
  int rank;
  float *overlaps = NULL;

  char trainfile_wrap[200];      /* file for warpped positive training examples */
  char trainfile_unwrap[200];    /* file for unwarpped positive training examples */
  char trainfile_negative[200];  /* file for negative training examples */
  char cadfile[200];             /* file with cad models */
  char modelfile[200];           /* file for resulting classifier */

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* select GPU */
  select_gpu(rank);

  /* choose a random seed */
  srand(time(NULL));

  svm_struct_learn_api_init(argc,argv);

  read_input_parameters(argc, argv, trainfile_wrap, trainfile_unwrap, trainfile_negative, cadfile, modelfile, &verbosity, &struct_verbosity, &struct_parm, &learn_parm, &kernel_parm, &alg_type);

  /* read cad models */
  cads = read_cad_model(cadfile, &cad_num, 0, &struct_parm);

  /* set the cad model for structmodel */
  structmodel.cad_num = cad_num;
  structmodel.cads = cads;

  if(struct_parm.hard_negative > 0)
  {
    printf("Read negative training samples for data mining\n");
    sample_negative = read_struct_examples_batch(trainfile_negative, &struct_parm, &structmodel);
    printf("Read negative training samples for data mining done\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  struct_parm.confile_read[0] = '\0';    /* start with empty constraint */
  struct_parm.epsilon_init = 100.0;

  /***************************************************/
  /* training with wrapped positives */
  /***************************************************/
  struct_parm.is_wrap = 1;
  if(struct_parm.full_model == 1)
    struct_parm.deep = 1;
  else
    struct_parm.deep = 0;

  /* intialize the overlaps */
  if(rank == 0)
  {
    fp = fopen(trainfile_unwrap, "r");
    if(fp == NULL)
    {
      printf("Cannot open file %s to read\n", trainfile_unwrap);
      exit(1);
    }
    fscanf(fp, "%d", &num);
    fclose(fp);
    overlaps = (float*)my_malloc(sizeof(float)*num);
    memset(overlaps, 0, sizeof(float)*num);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  sprintf(struct_parm.confile_write, "%s_wrap.con", struct_parm.cls);
  sprintf(filename, "%s_wrap.mod", struct_parm.cls);
  sprintf(filename_data, "tmp/wrap.dat");

  if(struct_parm.is_continuous == 1 && is_file_exist(struct_parm.confile_write) == 1 && is_file_exist(filename) == 1)
  {
    printf("%s exists\n", filename);
    strcpy(struct_parm.confile_read, struct_parm.confile_write);
    strcpy(modelfile, filename);
  }
  else
  {
    struct_parm.is_continuous = 0;
    printf("Train model %s\n", filename);

    /* contruct training data */
    if(rank == 0)
      random_negative_samples(filename_data, trainfile_wrap, trainfile_unwrap, trainfile_negative, modelfile, struct_parm.is_wrap, overlaps, cad_num, cads);
    MPI_Barrier(MPI_COMM_WORLD);

    /* read the training examples */
    sample = read_struct_examples(filename_data, &struct_parm, &structmodel);
    printf("Read training samples from %s done\n", filename_data);

    if(alg_type == 1)
      svm_learn_struct(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel);
    else if(alg_type == 2)
      svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, PRIMAL_ALG);
    else if(alg_type == 3)
      svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, DUAL_ALG);
    else if(alg_type == 4)
      svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, DUAL_CACHE_ALG);
    else
      exit(1);
      
    strcpy(modelfile, filename);
    if(rank == 0)
    {
      write_struct_model(modelfile, &structmodel, &struct_parm);
      printf("Train model with wrapped positives done\n");
    }
    strcpy(struct_parm.confile_read, struct_parm.confile_write);
    MPI_Barrier(MPI_COMM_WORLD);

    free_struct_model(structmodel);
    free_struct_sample(sample);
  }

  /***************************************************/
  /* training full model with latent positives */
  /***************************************************/
  struct_parm.is_wrap = 0;
  for(iter = 0; iter < struct_parm.latent_positive; iter++)
  {
    sprintf(struct_parm.confile_write, "%s_latent_%d.con", struct_parm.cls, iter);
    sprintf(filename, "%s_latent_%d.mod", struct_parm.cls, iter);
    sprintf(filename_data, "tmp/latent_%d.dat", iter);

    if(struct_parm.is_continuous == 1 && is_file_exist(struct_parm.confile_write) == 1 && is_file_exist(filename) == 1)
    {
      printf("%s exists\n", filename);
      strcpy(struct_parm.confile_read, struct_parm.confile_write);
      strcpy(modelfile, filename);
    }
    else
    {
      struct_parm.is_continuous = 0;
      printf("Train model %s\n", filename);

      /* contruct training data */
      if(rank == 0)
        random_negative_samples(filename_data, trainfile_wrap, trainfile_unwrap, trainfile_negative, modelfile, struct_parm.is_wrap, overlaps, cad_num, cads);
      MPI_Barrier(MPI_COMM_WORLD);

      /* read the training examples */
      sample = read_struct_examples(filename_data, &struct_parm, &structmodel);
      printf("Read training samples from %s done\n", filename_data);

      if(alg_type == 1)
        svm_learn_struct(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel);
      else if(alg_type == 2)
        svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, PRIMAL_ALG);
      else if(alg_type == 3)
        svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, DUAL_ALG);
      else if(alg_type == 4)
        svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, DUAL_CACHE_ALG);
      else
        exit(1);
      
      strcpy(modelfile, filename);
      if(rank == 0)
      {
        write_struct_model(modelfile, &structmodel, &struct_parm);
        printf("Train model with latent positives iter %d/%d done\n", iter, struct_parm.latent_positive);
      }
      strcpy(struct_parm.confile_read, struct_parm.confile_write);
      MPI_Barrier(MPI_COMM_WORLD);

      free_struct_model(structmodel);
      free_struct_sample(sample);
    }
  }

  /******************************************************/
  /* training with data mining hard examples */
  /******************************************************/
  struct_parm.is_wrap = 0;
  for(iter = 0; iter < struct_parm.hard_negative; iter++)
  {
    /* file name for constraints and model */
    sprintf(struct_parm.confile_write, "%s_hard_%d.con", struct_parm.cls, iter);
    sprintf(filename, "%s_hard_%d.mod", struct_parm.cls, iter);
    sprintf(filename_data, "tmp/hard_%d.dat", iter);

    if(struct_parm.is_continuous == 1 && is_file_exist(struct_parm.confile_write) == 1 && is_file_exist(filename) == 1)
    {
      printf("%s exists\n", filename);
      strcpy(struct_parm.confile_read, struct_parm.confile_write);
      strcpy(modelfile, filename);
    }
    else
    {
      struct_parm.is_continuous = 0;
      printf("Train model %s\n", filename);

      /* latent positive and data mining hard negative */
      if(rank == 0)
        random_negative_samples("tmp/temp.dat", trainfile_wrap, trainfile_unwrap, trainfile_negative, modelfile, struct_parm.is_wrap, overlaps, cad_num, cads);
      MPI_Barrier(MPI_COMM_WORLD);
      data_mining_hard_examples(filename_data, "tmp/temp.dat", trainfile_negative, modelfile, sample_negative, cad_num, cads);
      
      /* read the training examples */
      sample = read_struct_examples(filename_data, &struct_parm, &structmodel);
      printf("Hard data mining %d/%d: read training examples from %s done\n", iter, struct_parm.hard_negative, filename_data);
    
      if(alg_type == 1)
        svm_learn_struct(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel);
      else if(alg_type == 2)
        svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, PRIMAL_ALG);
      else if(alg_type == 3)
        svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, DUAL_ALG);
      else if(alg_type == 4)
        svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, DUAL_CACHE_ALG);
      else
        exit(1);
      
      strcpy(modelfile, filename);
      if(rank == 0)
      {
        write_struct_model(modelfile, &structmodel, &struct_parm);
        printf("Train model hard iter %d/%d done\n", iter, struct_parm.hard_negative);
      }
      flag = is_confile_same(struct_parm.confile_read, struct_parm.confile_write);
      strcpy(struct_parm.confile_read, struct_parm.confile_write);
      MPI_Barrier(MPI_COMM_WORLD);
        
      /* free the structmodel and samples */
      free_struct_model(structmodel);
      free_struct_sample(sample);
      if(flag == 1)
        break;
    }
  }

  if(rank == 0)
  {
    structmodel = read_struct_model(modelfile, &struct_parm);
    sprintf(modelfile, "%s_final.mod", struct_parm.cls);
    write_struct_model(modelfile, &structmodel, &struct_parm);
    free_struct_model(structmodel);
    free(overlaps);
    printf("Write the final model done\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  free_struct_sample(sample_negative);

  for(i = 0; i < cad_num; i++)
    destroy_cad(cads[i]);
  free(cads);

  svm_struct_learn_api_exit();

  MPI_Finalize();

  return 0;
}

/* check if file exists or not */
int is_file_exist(char *filename)
{
  FILE *fp;
  fp = fopen(filename, "r");
  if(fp == NULL)
    return 0;
  else
  {
    fclose(fp);
    return 1;
  }
}

/* check if no new constraint added */
int is_confile_same(char* confile_read, char* confile_write)
{
  int num_read, num_write;
  FILE *fp;

  /* read number of constraints of confile_read */
  fp = fopen(confile_read, "r");
  if(fp == NULL)
    return 0;
  else
  {
    fscanf(fp, "%d", &num_read);
    fclose(fp);
  }
  
  /* read number of constraints of confile_write */
  fp = fopen(confile_write, "r");
  if(fp == NULL)
    return 0;
  else
  {
    fscanf(fp, "%d", &num_write);
    fclose(fp);
  }

  if(num_read == num_write)
    return 1;
  else
    return 0;
}

/* randomly sample negative samples */
/* only run in one process */
void random_negative_samples(char *filename, char *trainfile_wrap, char *trainfile_unwrap, char *trainfile_negative, char *modelfile, int is_wrap, float *overlaps, int cad_num, CAD **cads)
{
  char line[BUFFLE_SIZE];
  int i, index, num_pos, num_pos_used, num_neg, num_neg_used;
  int object_label;
  int *flag;
  float overlap, threshold = 0.6;
  FILE *fp, *fp_prev, *fp_wrap, *fp_unwrap, *fp_negative;
  STRUCTMODEL sm;
  STRUCT_LEARN_PARM sparm;
  LABEL* y;
  EXAMPLE example;
  CUMATRIX* matrix;

  /* number of positive samples */
  fp_wrap = fopen(trainfile_wrap, "r");
  if(fp_wrap == NULL)
  {
    printf("Cannot open file %s to read\n", trainfile_wrap);
    exit(1);
  }
  fscanf(fp_wrap, "%d\n", &num_pos);

  fp_unwrap = fopen(trainfile_unwrap, "r");
  if(fp_unwrap == NULL)
  {
    printf("Cannot open file %s to read\n", trainfile_unwrap);
    exit(1);
  }
  fscanf(fp_unwrap, "%d\n", &num_pos);

  /* number of negative samples */
  fp_negative = fopen(trainfile_negative, "r");
  if(fp_negative == NULL)
  {
    printf("Cannot open file %s to read\n", trainfile_negative);
    exit(1);
  }
  fscanf(fp_negative, "%d\n", &num_neg);

  /* construct new training data and write to file */
  printf("Writing data to %s\n", filename);
  fp = fopen(filename, "w");

  /* determine the number of positives to be used */
  if(is_wrap)
    num_pos_used = num_pos;
  else
  {
    /* read model */
    sm = read_struct_model(modelfile, &sparm);
    sm.cad_num = cad_num;
    sm.cads = cads;
    num_pos_used = 0;
    y = (LABEL*)my_malloc(sizeof(LABEL)*num_pos);
    memset(y, 0, sizeof(LABEL)*num_pos);
    matrix = (CUMATRIX*)my_malloc(sizeof(CUMATRIX)*num_pos);
    memset(matrix, 0, sizeof(CUMATRIX)*num_pos);
    flag = (int*)my_malloc(sizeof(int)*num_pos);
    memset(flag, 0, sizeof(int)*num_pos);

    for(i = 0; i < num_pos; i++)
    {
      example = read_struct_example_one(fp_unwrap, &(matrix[i]), &sparm, &sm);
      y[i] = find_most_positive_constraint(example.x, example.y, &sm, &sparm);
      overlap = box_overlap(y[i].bbox, example.y.bbox);
      printf("Latent positive example %d/%d: overlap %.2f, previous max overlap %.2f\n", i+1, num_pos, overlap, overlaps[i]);
      if(overlap > threshold)
      {
        num_pos_used++;
        flag[i] = 1;
      }
      if(overlap > overlaps[i])
        overlaps[i] = overlap;

      free_pattern(example.x);
      free_label(example.y);
    }
  }  

  num_neg_used = TRAIN_NEG_NUM > num_neg/2 ? num_neg/2 : TRAIN_NEG_NUM;
  fprintf(fp, "%d\n", num_pos_used + num_neg_used);

  /* write positive samples */
  if(is_wrap)  /* use wrapped positives */
  {
    printf("Use %d wrapped positives\n", num_pos);
    for(i = 0; i < num_pos; i++)
    {
      fgets(line, BUFFLE_SIZE, fp_wrap);
      sscanf(line, "%d", &object_label);
      if(object_label == 1)
        fputs(line, fp);
      else
      {
        printf("Error in read wrapped positive example %d\n", i);
        exit(1);
      }
    }
    fclose(fp_wrap);
    fclose(fp_unwrap);
  }
  else  /* use latent positives */
  {
    printf("Use %d latent positives\n", num_pos_used);
    for(i = 0; i < num_pos; i++)
    {
      if(flag[i] == 1)
        write_latent_positive(fp, y[i], matrix[i], &sm, &sparm);

      free_label(y[i]);
      free_cumatrix(&(matrix[i]));
    }
    free(y);
    free(matrix);
    free(flag);
    fclose(fp_wrap);
    fclose(fp_unwrap);
  }

  /* randomly sample num_neg_used negative samples */
  flag = (int*)my_malloc(sizeof(int)*num_neg);
  memset(flag, 0, sizeof(int)*num_neg);
  i = 0;
  while(i < num_neg_used)
  {
    index = (int)rand() % num_neg;
    if(flag[index] == 0)
    {
      flag[index] = 1;
      i++;
    }
  }

  /* write negative samples */
  for(i = 0; i < num_neg; i++)
  {
    fgets(line, BUFFLE_SIZE, fp_negative);
    if(flag[i])
      fputs(line, fp);
  }
  fclose(fp_negative);
  fclose(fp);

  /* clean up */
  free(flag);
  if(!is_wrap)
    free_struct_model(sm);
}


/* data mining hard negative samples */
void data_mining_hard_examples(char *filename, char *trainfile, char *testfile, char *modelfile, SAMPLE testsample, int cad_num, CAD **cads)
{
  int i, j, n, ntrain, ntest, num, object_label, count, count_pos, count_neg;
  char line[BUFFLE_SIZE];
  double *energy, *energies;
  LABEL *y = NULL;
  ENERGYINDEX *energy_index;
  FILE *fp, *ftrain, *ftest;
  STRUCTMODEL sm;
  STRUCT_LEARN_PARM sparm;

  /* MPI process */
  int rank;
  int procs_num;
  int start, end, block_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs_num);

  /* read model */
  sm = read_struct_model(modelfile, &sparm);
  sm.cad_num = cad_num;
  sm.cads = cads;
  printf("Hard data mining negative samples, read modelfile %s done\n", modelfile);

  n = testsample.n;
  block_size = (n+procs_num-1) / procs_num;
  start = rank*block_size;
  end = start+block_size-1 > n-1 ? n-1 : start+block_size-1;

  energy = (double*)my_malloc(sizeof(double)*n);
  energies = (double*)my_malloc(sizeof(double)*n);
  memset(energy, 0, sizeof(double)*n);
  memset(energies, 0, sizeof(double)*n);
  
  for(i = start; i <= end; i++) 
  {
    y = classify_struct_example(testsample.examples[i].x, &num, 0, &sm, &sparm);
    count = 0;
    for(j = 0; j < num; j++)
    {
      if(y[j].object_label)
        count++;
    }
    printf("Data mining hard negative example %d/%d: %d objects detected\n", i+1, n, count);
    if(count == 0)
      energy[i] = -5*sparm.loss_value;
    else
    {
      for(j = 0; j < num; j++)
      {
        if(y[j].object_label)
        {
          energy[i] = y[j].energy;
          break;
        }
      }
    }
    /* free labels */
    for(j = 0; j < num; j++)
      free_label(y[j]);
    free(y);
  }
  MPI_Allreduce(energy, energies, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if(rank == 0)
  {
    energy_index = (ENERGYINDEX*)my_malloc(sizeof(ENERGYINDEX)*n);
    for(i = 0; i < n; i++)
    {
      energy_index[i].index = i;
      energy_index[i].energy = energies[i];
    }
    /* sort energies */
    qsort(energy_index, n, sizeof(ENERGYINDEX), compare_energy);

    /* construct new training data and write to file */
    printf("Writing data to %s\n", filename);
    fp = fopen(filename, "w");
    ftrain = fopen(trainfile, "r");
    if(ftrain == NULL)
    {
      printf("Cannot open file %s to read\n", trainfile);
      exit(1);
    }
    ftest = fopen(testfile, "r");
    if(ftest == NULL)
    {
      printf("Cannot open file %s to read\n", testfile);
      exit(1);
    }

    /* positive samples from training data file */
    fscanf(ftrain, "%d\n", &ntrain);
    fscanf(ftest, "%d\n", &ntest);

    fprintf(fp, "%d\n", ntrain);
    count_pos = 0;
    for(i = 0; i < ntrain; i++)
    {
      fgets(line, BUFFLE_SIZE, ftrain);
      sscanf(line, "%d", &object_label);
      if(object_label == 1)
      {
        fputs(line, fp);
        count_pos++;
      }
      else
        break;
    }
    fclose(ftrain);
    count_neg = ntrain - count_pos;

    /* negative samples from hard negative examples */
    count = 0;
    for(i = 0; i < ntest; i++)
    {
      fgets(line, BUFFLE_SIZE, ftest);
      for(j = 0; j < count_neg; j++)
      {
        if(i == energy_index[j].index)
        {
          count++;
          fputs(line, fp);
          printf("Use negative example %d: energy %.2f\n", i, energy_index[j].energy);
          break;
        }
      }
      if(count >= count_neg)
        break;
    }
    fclose(ftest);
    fclose(fp);
    free(energy_index);
    printf("Done\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  free(energy);
  free(energies);
  free_struct_model(sm);
}

int compare_energy(const void *a, const void *b)
{
  double diff;
  diff =  ((ENERGYINDEX*)a)->energy - ((ENERGYINDEX*)b)->energy;
  if(diff < 0)
    return 1;
  else if(diff > 0)
    return -1;
  else
    return 0;
}

/*---------------------------------------------------------------------------*/

void read_input_parameters(int argc, char *argv[], char *trainfile_wrap, char *trainfile_unwrap, char *trainfile_negative, char *cadfile, char *modelfile,
			   long *verbosity,long *struct_verbosity, 
			   STRUCT_LEARN_PARM *struct_parm,
			   LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm,
			   int *alg_type)
{
  int len;
  long i;
  char type[100];
  
  /* set default */
  (*alg_type)=DEFAULT_ALG_TYPE;
  struct_parm->C=-0.01;
  struct_parm->slack_norm=1;
  struct_parm->epsilon=DEFAULT_EPS;
  struct_parm->epsilon_init = 100.0;
  struct_parm->custom_argc=0;
  struct_parm->loss_function=DEFAULT_LOSS_FCT;
  struct_parm->loss_type=DEFAULT_RESCALING;
  struct_parm->newconstretrain=100;
  struct_parm->ccache_size=5;

  /* loss parameters */
  struct_parm->loss_value = 100;
  struct_parm->wpair = 10;
  struct_parm->hard_negative = 5;
  struct_parm->latent_positive = 5;
  struct_parm->is_continuous = 1;
  struct_parm->fractions[0] = 1.0;
  struct_parm->iter = 0;
  struct_parm->deep = 1;
  struct_parm->full_model = 1;
  struct_parm->is_wrap = 1;

  struct_parm->cset.lhs = NULL;
  struct_parm->cset.rhs = NULL;
  struct_parm->cset.m = 0;

  strcpy (modelfile, "svm_struct_model");
  strcpy (learn_parm->predfile, "trans_predictions");
  strcpy (learn_parm->alphafile, "");
  (*verbosity)=0;/*verbosity for svm_light*/
  (*struct_verbosity)=1; /*verbosity for struct learning portion*/
  learn_parm->biased_hyperplane = 0;
  learn_parm->remove_inconsistent=0;
  learn_parm->skip_final_opt_check=0;
  learn_parm->svm_maxqpsize=10;
  learn_parm->svm_newvarsinqp=0;
  learn_parm->svm_iter_to_shrink=-9999;
  learn_parm->maxiter=100000;
  learn_parm->kernel_cache_size=40;
  learn_parm->svm_c=99999999;  /* overridden by struct_parm->C */
  learn_parm->eps=0.001;       /* overridden by struct_parm->epsilon */
  learn_parm->transduction_posratio=-1.0;
  learn_parm->svm_costratio=1.0;
  learn_parm->svm_costratio_unlab=1.0;
  learn_parm->svm_unlabbound=1E-5;
  learn_parm->epsilon_crit=0.001;
  learn_parm->epsilon_a=1E-10;  /* changed from 1e-15 */
  learn_parm->compute_loo=0;
  learn_parm->rho=1.0;
  learn_parm->xa_depth=0;
  kernel_parm->kernel_type=0;
  kernel_parm->poly_degree=3;
  kernel_parm->rbf_gamma=1.0;
  kernel_parm->coef_lin=1;
  kernel_parm->coef_const=1;
  strcpy(kernel_parm->custom,"empty");
  strcpy(type,"c");

  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) 
      { 
      case '?': print_help(); exit(0);
      case 'a': i++; strcpy(learn_parm->alphafile,argv[i]); break;
      case 'c': i++; struct_parm->C=atof(argv[i]); break;
      case 'p': i++; struct_parm->slack_norm=atol(argv[i]); break;
      case 'e': i++; struct_parm->epsilon=atof(argv[i]); break;
      case 'k': i++; struct_parm->newconstretrain=atol(argv[i]); break;
      case 'h': i++; learn_parm->svm_iter_to_shrink=atol(argv[i]); break;
      case '#': i++; learn_parm->maxiter=atol(argv[i]); break;
      case 'm': i++; learn_parm->kernel_cache_size=atol(argv[i]); break;
      case 'w': i++; (*alg_type)=atol(argv[i]); break;
      case 'o': i++; struct_parm->loss_type=atol(argv[i]); break;
      case 'n': i++; learn_parm->svm_newvarsinqp=atol(argv[i]); break;
      case 'q': i++; learn_parm->svm_maxqpsize=atol(argv[i]); break;
      case 'l': i++; struct_parm->loss_function=atol(argv[i]); break;
      case 'f': i++; struct_parm->ccache_size=atol(argv[i]); break;
      case 't': i++; kernel_parm->kernel_type=atol(argv[i]); break;
      case 'd': i++; kernel_parm->poly_degree=atol(argv[i]); break;
      case 'g': i++; kernel_parm->rbf_gamma=atof(argv[i]); break;
      case 's': i++; kernel_parm->coef_lin=atof(argv[i]); break;
      case 'r': i++; kernel_parm->coef_const=atof(argv[i]); break;
      case 'u': i++; strcpy(kernel_parm->custom,argv[i]); break;
      case '-': strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);i++; strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);break; 
      case 'v': i++; (*struct_verbosity)=atol(argv[i]); break;
      case 'y': i++; (*verbosity)=atol(argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n",argv[i]);
	       print_help();
	       exit(0);
      }
  }
  if(i >= argc) 
  {
    printf("\nNot enough input parameters!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  strcpy (trainfile_wrap, argv[i]);
  i++;
  strcpy (trainfile_unwrap, argv[i]);
  i++;
  strcpy (trainfile_negative, argv[i]);
  i++;
  strcpy (cadfile, argv[i]);

  /* construct class name */
  len = strlen(cadfile);
  strncpy(struct_parm->cls, cadfile, len-4);
  struct_parm->cls[len-4] = '\0';

  if(learn_parm->svm_iter_to_shrink == -9999) {
    learn_parm->svm_iter_to_shrink=100;
  }

  if((learn_parm->skip_final_opt_check) 
     && (kernel_parm->kernel_type == LINEAR)) {
    printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
    learn_parm->skip_final_opt_check=0;
  }    
  if((learn_parm->skip_final_opt_check) 
     && (learn_parm->remove_inconsistent)) {
    printf("\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
    wait_any_key();
    print_help();
    exit(0);
  }    
  if((learn_parm->svm_maxqpsize<2)) {
    printf("\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n",learn_parm->svm_maxqpsize); 
    wait_any_key();
    print_help();
    exit(0);
  }
  if((learn_parm->svm_maxqpsize<learn_parm->svm_newvarsinqp)) {
    printf("\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n",learn_parm->svm_maxqpsize); 
    printf("new variables [%ld] entering the working set in each iteration.\n",learn_parm->svm_newvarsinqp); 
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->svm_iter_to_shrink<1) {
    printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",learn_parm->svm_iter_to_shrink);
    wait_any_key();
    print_help();
    exit(0);
  }
  if(struct_parm->C<0) {
    printf("\nYou have to specify a value for the parameter '-c' (C>0)!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(((*alg_type) < 1) || ((*alg_type) > 4)) {
    printf("\nAlgorithm type must be either '1', '2', '3', or '4'!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->transduction_posratio>1) {
    printf("\nThe fraction of unlabeled examples to classify as positives must\n");
    printf("be less than 1.0 !!!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->svm_costratio<=0) {
    printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(struct_parm->epsilon<=0) {
    printf("\nThe epsilon parameter must be greater than zero!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if((struct_parm->slack_norm<1) || (struct_parm->slack_norm>2)) {
    printf("\nThe norm of the slacks must be either 1 (L1-norm) or 2 (L2-norm)!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if((struct_parm->loss_type != SLACK_RESCALING) 
     && (struct_parm->loss_type != MARGIN_RESCALING)) {
    printf("\nThe loss type must be either 1 (slack rescaling) or 2 (margin rescaling)!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if(learn_parm->rho<0) {
    printf("\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
    printf("be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
    printf("Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
    wait_any_key();
    print_help();
    exit(0);
  }
  if((learn_parm->xa_depth<0) || (learn_parm->xa_depth>100)) {
    printf("\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
    printf("for switching to the conventional xa/estimates described in T. Joachims,\n");
    printf("Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
    wait_any_key();
    print_help();
    exit(0);
  }

  parse_struct_parameters(struct_parm);
}

void wait_any_key()
{
  printf("\n(more)\n");
  (void)getc(stdin);
}

void print_help()
{
  printf("\nSVM-struct learning module: %s, %s, %s\n",INST_NAME,INST_VERSION,INST_VERSION_DATE);
  printf("   includes SVM-struct %s for learning complex outputs, %s\n",STRUCT_VERSION,STRUCT_VERSION_DATE);
  printf("   includes SVM-light %s quadratic optimizer, %s\n",VERSION,VERSION_DATE);
  copyright_notice();
  printf("   usage: svm_struct_learn [options] example_file model_file\n\n");
  printf("Arguments:\n");
  printf("         example_file-> file with training data\n");
  printf("         model_file  -> file to store learned decision rule in\n");

  printf("General options:\n");
  printf("         -?          -> this help\n");
  printf("         -v [0..3]   -> verbosity level (default 1)\n");
  printf("         -y [0..3]   -> verbosity level for svm_light (default 0)\n");
  printf("Learning options:\n");
  printf("         -c float    -> C: trade-off between training error\n");
  printf("                        and margin (default 0.01)\n");
  printf("         -p [1,2]    -> L-norm to use for slack variables. Use 1 for L1-norm,\n");
  printf("                        use 2 for squared slacks. (default 1)\n");
  printf("         -o [1,2]    -> Rescaling method to use for loss.\n");
  printf("                        1: slack rescaling\n");
  printf("                        2: margin rescaling\n");
  printf("                        (default %d)\n",DEFAULT_RESCALING);
  printf("         -l [0..]    -> Loss function to use.\n");
  printf("                        0: zero/one loss\n");
  printf("                        (default %d)\n",DEFAULT_LOSS_FCT);
  printf("Kernel options:\n");
  printf("         -t int      -> type of kernel function:\n");
  printf("                        0: linear (default)\n");
  printf("                        1: polynomial (s a*b+c)^d\n");
  printf("                        2: radial basis function exp(-gamma ||a-b||^2)\n");
  printf("                        3: sigmoid tanh(s a*b + c)\n");
  printf("                        4: user defined kernel from kernel.h\n");
  printf("         -d int      -> parameter d in polynomial kernel\n");
  printf("         -g float    -> parameter gamma in rbf kernel\n");
  printf("         -s float    -> parameter s in sigmoid/poly kernel\n");
  printf("         -r float    -> parameter c in sigmoid/poly kernel\n");
  printf("         -u string   -> parameter of user defined kernel\n");
  printf("Optimization options (see [2][3]):\n");
  printf("         -w [1,2,3,4]-> choice of structural learning algorithm (default %d):\n",(int)DEFAULT_ALG_TYPE);
  printf("                        1: algorithm described in [2]\n");
  printf("                        2: joint constraint algorithm (primal) [to be published]\n");
  printf("                        3: joint constraint algorithm (dual) [to be published]\n");
  printf("                        4: joint constraint algorithm (dual) with constr. cache\n");
  printf("         -q [2..]    -> maximum size of QP-subproblems (default 10)\n");
  printf("         -n [2..q]   -> number of new variables entering the working set\n");
  printf("                        in each iteration (default n = q). Set n<q to prevent\n");
  printf("                        zig-zagging.\n");
  printf("         -m [5..]    -> size of cache for kernel evaluations in MB (default 40)\n");
  printf("                        (used only for -w 1 with kernels)\n");
  printf("         -f [5..]    -> number of constraints to cache for each example\n");
  printf("                        (default 5) (used with -w 4)\n");
  printf("         -e float    -> eps: Allow that error for termination criterion\n");
  printf("                        (default %f)\n",DEFAULT_EPS);
  printf("         -h [5..]    -> number of iterations a variable needs to be\n"); 
  printf("                        optimal before considered for shrinking (default 100)\n");
  printf("         -k [1..]    -> number of new constraints to accumulate before\n"); 
  printf("                        recomputing the QP solution (default 100) (-w 1 only)\n");
  printf("         -# int      -> terminate QP subproblem optimization, if no progress\n");
  printf("                        after this number of iterations. (default 100000)\n");
  printf("Output options:\n");
  printf("         -a string   -> write all alphas to this file after learning\n");
  printf("                        (in the same order as in the training set)\n");
  printf("Structure learning options:\n");
  print_struct_help();
  wait_any_key();

  printf("\nMore details in:\n");
  printf("[1] T. Joachims, Learning to Align Sequences: A Maximum Margin Aproach.\n");
  printf("    Technical Report, September, 2003.\n");
  printf("[2] I. Tsochantaridis, T. Joachims, T. Hofmann, and Y. Altun, Large Margin\n");
  printf("    Methods for Structured and Interdependent Output Variables, Journal\n");
  printf("    of Machine Learning Research (JMLR), Vol. 6(Sep):1453-1484, 2005.\n");
  printf("[3] T. Joachims, Making Large-Scale SVM Learning Practical. Advances in\n");
  printf("    Kernel Methods - Support Vector Learning, B. Sch\F6lkopf and C. Burges and\n");
  printf("    A. Smola (ed.), MIT Press, 1999.\n");
  printf("[4] T. Joachims, Learning to Classify Text Using Support Vector\n");
  printf("    Machines: Methods, Theory, and Algorithms. Dissertation, Kluwer,\n");
  printf("    2002.\n\n");
}
