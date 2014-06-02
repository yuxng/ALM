/***********************************************************************/
/*                                                                     */
/*   svm_struct_learn.c                                                */
/*                                                                     */
/*   Basic algorithm for learning structured outputs (e.g. parses,     */
/*   sequences, multi-label classification) with a Support Vector      */ 
/*   Machine.                                                          */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 26.06.06                                                    */
/*   Modified by: Yu Xiang                                             */
/*   Date: 05.01.12                                                    */
/*                                                                     */
/*   Copyright (c) 2006  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include "svm_struct_learn.h"
#include "svm_struct_common.h"
#include "svm_struct_api.h"
#include <mpi.h>
#include <assert.h>

#define MAX(x,y)      ((x) < (y) ? (y) : (x))
#define MIN(x,y)      ((x) > (y) ? (y) : (x))


void svm_learn_struct(SAMPLE sample, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm, STRUCTMODEL *sm)
{
  int         i, j, flag_terminate;
  int         numIt = 0;
  long        argmax_count = 0;
  long        newconstraints = 0, totconstraints = 0, activenum = 0; 
  int         opti_round, *opti, fullround;
  long        old_numConst = 0, init_slack = 0;
  double      epsilon, svmCnorm;
  long        tolerance, new_precision = 1;
  double      lossval, factor, dist;
  double      margin = 0;
  double      slack, *slacks, slacksum, ceps;
  long        sizePsi;
  double      *alpha = NULL;
  long        *alphahist = NULL, optcount = 0, lastoptcount = 0;
  CONSTSET    cset;
  SVECTOR     *diff = NULL;
  SVECTOR     *fy, *fybar, *f;
  SVECTOR     *slackvec;
  WORD        slackv[2];
  MODEL       *svmModel = NULL;
  KERNEL_CACHE *kcache = NULL;
  LABEL       ybar;
  DOC         *doc;

  long        n = sample.n;
  EXAMPLE     *ex = sample.examples;
  double      rt_total = 0, rt_opt = 0, rt_init = 0, rt_psi = 0, rt_viol = 0;
  double      rt1, rt2;
  double y_score, ybar_score, azimuth, elevation, distance, prediction, cur_loss, energy;
  double *diff_margin_energy = NULL, *diff_n = NULL;

  /* MPI process */
  int rank;
  int procs_num;
  MPI_Status Stat;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs_num);

  rt1 = get_runtime();

  init_struct_model(sample, sm, sparm, lparm, kparm); 
  sizePsi = sm->sizePsi+1;          /* sm must contain size of psi on return */

  /* initialize example selection heuristic */ 
  opti = (int*)my_malloc(n * sizeof(int));
  for(i = 0; i < n; i++)
  {
    opti[i] = 0;
  }
  opti_round = 0;

  /* for MPI usage */
  diff_n = create_nvector(sm->sizePsi);
  clear_nvector(diff_n, sm->sizePsi);
  diff_margin_energy = create_nvector(sm->sizePsi+2);
  clear_nvector(diff_margin_energy, sm->sizePsi+2);

  /* normalize regularization parameter C by the number of training examples */
  svmCnorm = sparm->C / n;

  if(sparm->slack_norm == 1)
  {
    lparm->svm_c = svmCnorm;          /* set upper bound C */
    lparm->sharedslack = 1;
  }
  else if(sparm->slack_norm == 2)
  {
    lparm->svm_c = 999999999999999.0; /* upper bound C must never be reached */
    lparm->sharedslack = 0;
    if(kparm->kernel_type != LINEAR) 
    {
      printf("ERROR: Kernels are not implemented for L2 slack norm!"); 
      fflush(stdout);
      exit(0); 
    }
  }
  else
  {
    printf("ERROR: Slack norm must be L1 or L2!");
    fflush(stdout);
    exit(0);
  }

  tolerance = MIN(n/3,MAX(n/100,5));/* increase precision, whenever less than that number of constraints is not fulfilled */
  lparm->biased_hyperplane = 0;     /* set threshold to zero */

  if(rank == 0)
  {
    cset = init_struct_constraints(sample, &alpha, sm, sparm);
    init_slack = 0;
    if(cset.m > 0)
    {
      alphahist = (long*)realloc(alphahist, sizeof(long)*cset.m);
      for(i = 0; i < cset.m; i++)
      {
        alphahist[i] = -1; /* -1 makes sure these constraints are never removed */
        if(cset.lhs[i]->slackid > init_slack)
          init_slack = cset.lhs[i]->slackid;
      }
    }
    printf("initial slack index = %d\n", init_slack);
  }
  MPI_Bcast(&(sparm->epsilon_init), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  epsilon = sparm->epsilon_init;                  /* start with low precision and increase later */
  MPI_Barrier(MPI_COMM_WORLD);

  /* set initial model and slack variables*/
  svmModel = (MODEL *)my_malloc(sizeof(MODEL));
  memset(svmModel, 0, sizeof(MODEL));

  if(rank == 0)
  { 
    lparm->epsilon_crit = epsilon;
    if(kparm->kernel_type != LINEAR)
      kcache = kernel_cache_init(MAX(cset.m, 1), lparm->kernel_cache_size);
    svm_learn_optimization_linear(cset.lhs, cset.rhs, cset.m, sizePsi + n, init_slack+n, lparm, kparm, kcache, svmModel, alpha, 1);
    if(kcache)
      kernel_cache_cleanup(kcache);
    if(kparm->kernel_type != LINEAR)
      add_weight_vector_to_linear_model(svmModel);
  }
  MPI_Bcast(&(svmModel->totwords), 1, MPI_LONG, 0, MPI_COMM_WORLD);

  if(rank != 0)
  {
    svmModel->kernel_parm.kernel_type = kparm->kernel_type;
    svmModel->lin_weights = create_nvector(svmModel->totwords);
    clear_nvector(svmModel->lin_weights, svmModel->totwords);
  }
  /* broadcast the weights */
  MPI_Bcast(svmModel->lin_weights, svmModel->totwords+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(svmModel->b), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  sm->svm_model = svmModel;
  sm->w = svmModel->lin_weights; /* short cut to weight vector */
  copy_to_float_weights(sm);

  rt_init += MAX(get_runtime()-rt1,0);
  rt_total += MAX(get_runtime()-rt1,0);

  /*****************/
  /*** main loop ***/
  /*****************/
  do
  { /* iteratively increase precision */
    /*
    epsilon = MAX(epsilon*0.49999999999, sparm->epsilon);
    */
    epsilon = sparm->epsilon;
    new_precision = 1;
    if(epsilon == sparm->epsilon)   /* for final precision, find all SV */
      tolerance = MAX(0, round(n*0.01)); 
    lparm->epsilon_crit = epsilon/2;  /* svm precision must be higher than eps */
    if(rank == 0 && struct_verbosity>=1)
      printf("Setting current working precision to %g, tolerance = %ld\n", epsilon, tolerance);

    do
    { /* iteration until (approx) all SV are found for current precision and tolerance */
      opti_round++;
      activenum = n;

      do 
      { /* go through examples that keep producing new constraints */

        if(rank == 0 && struct_verbosity >= 1)
        { 
          printf("Iter %i (%ld active): ",++numIt,activenum); 
          fflush(stdout);
        }

        if(rank == 0)	
          old_numConst = cset.m;
        ceps = 0;
        fullround = (activenum == n);

        /*** example loop ***/
        for(i = 0; i < n; i++)
        {
          if(rank != 0 && i % procs_num != rank)
            continue;

          rt1 = get_runtime();
	    
          if(opti[i] != opti_round) /* if the example is not shrunk away, then see if it is necessary to add a new constraint */
          {
            if(rank == 0 && i % procs_num != rank)  /* receive data from the other process */
            {
              MPI_Recv(diff_margin_energy, sm->sizePsi+3, MPI_DOUBLE, i % procs_num, 1, MPI_COMM_WORLD, &Stat);
              fy = create_svector_n(diff_margin_energy, sm->sizePsi, "", 1.0);
              margin = diff_margin_energy[sm->sizePsi+1];
              energy = diff_margin_energy[sm->sizePsi+2];
            }
            else
            {
              rt2 = get_runtime();
              argmax_count++;
              if(sparm->loss_type == SLACK_RESCALING) 
                ybar=find_most_violated_constraint_slackrescaling(ex[i].x, ex[i].y, sm, sparm);
              else
                ybar=find_most_violated_constraint_marginrescaling(ex[i].x, ex[i].y, sm, sparm);
              energy = ybar.energy;
              rt_viol += MAX(get_runtime()-rt2,0);
	  
              /**** get psi(y)-psi(ybar) ****/
              rt2 = get_runtime();

              if(sparm->is_wrap)
                fy = psi(ex[i].x, ex[i].y, 0, sm, sparm);
              else
                fy = psi(ex[i].x, ex[i].y, 1, sm, sparm);

              doc = create_example(n, 0, i+1, 1, fy);
              y_score = classify_example(sm->svm_model, doc);
              free_example(doc, 0);

              fybar = psi(ex[i].x, ybar, 1, sm, sparm);

              doc = create_example(n, 0, i+1, 1, fybar);
              ybar_score = classify_example(sm->svm_model, doc);
              free_example(doc, 0);

              if(ybar.object_label == 1)
              {
                azimuth = sm->cads[ybar.cad_label]->objects2d[ybar.view_label]->azimuth;
                elevation = sm->cads[ybar.cad_label]->objects2d[ybar.view_label]->elevation;
                distance = sm->cads[ybar.cad_label]->objects2d[ybar.view_label]->distance;
                printf("example %d, (energy ybar %.2f, view %d, azimuth %.1f, elevation %.1f, distance %.1f) ", i+1, ybar.energy, ybar.view_label, azimuth, elevation, distance); fflush(stdout);
              }
              else
              {
                printf("example %d, ", i+1); fflush(stdout);
              }
              printf("score ybar %.2f, ", ybar_score); fflush(stdout);
              prediction = y_score - ybar_score;
              printf("score %.2f, ", prediction); fflush(stdout);

              rt_psi += MAX(get_runtime()-rt2, 0);

              lossval = loss(ex[i].y, ybar, sparm);
              free_label(ybar);

              printf("loss %.2f, ", lossval); fflush(stdout);

              /* calculate loss */
              cur_loss = lossval - prediction;
              if(cur_loss < 0.0)
                cur_loss = 0.0;
              printf("slack %.2f\n", cur_loss); fflush(stdout);
	    
      	      /**** scale feature vector and margin by loss ****/
              if(sparm->slack_norm == 2)
                lossval = sqrt(lossval);
              if(sparm->loss_type == SLACK_RESCALING)
                factor = lossval;
              else               /* do not rescale vector for */
                factor = 1.0;      /* margin rescaling loss type */

              for(f = fy; f; f = f->next)
                f->factor *= factor;
              for(f = fybar; f; f = f->next)
                f->factor *= -factor;
              margin = lossval;

              append_svector_list(fy, fybar);/* append the two vector lists */

              if(rank != 0)  /* MPI send data */
              {
                clear_nvector(diff_n, sm->sizePsi);
                add_list_n_ns(diff_n, fy, 1.0); /* add fy-fybar to sum */
                memcpy(diff_margin_energy, diff_n, sizeof(double)*(sm->sizePsi+1));
                diff_margin_energy[sm->sizePsi+1] = margin;
                diff_margin_energy[sm->sizePsi+2] = energy;

                MPI_Send(diff_margin_energy, sm->sizePsi+3, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                MPI_Recv(svmModel->lin_weights, svmModel->totwords+1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &Stat);
                MPI_Recv(&(svmModel->b), 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &Stat);
                sm->svm_model = svmModel;
                sm->w = svmModel->lin_weights;
                copy_to_float_weights(sm);
              }
            }

            if(rank == 0)
            {
              /**** create constraint for current ybar ****/
              doc = create_example(cset.m, 0, init_slack+i+1, 1, fy);

      	      /**** compute slack for this example ****/
              slack = 0;
              for(j = 0; j < cset.m; j++)
              {
                if(cset.lhs[j]->slackid == init_slack+i+1)
                {
                  if(sparm->slack_norm == 2) /* works only for linear kernel */
                    slack = MAX(slack, cset.rhs[j] - (classify_example(svmModel, cset.lhs[j]) - sm->w[sizePsi+i]/(sqrt(2*svmCnorm))));
                  else
                    slack=MAX(slack, cset.rhs[j] - classify_example(svmModel, cset.lhs[j]));
                }
              }
	    
              /**** if `error' add constraint and recompute ****/
              dist = classify_example(svmModel, doc);
              ceps = MAX(ceps, margin - dist - slack);
              /*
              if(slack > (margin-dist+0.0001))
              {
                printf("\nWARNING: Slack of most violated constraint is smaller than slack of working\n");
                printf("         set! There is probably a bug in 'find_most_violated_constraint_*'.\n");
                printf("Ex %d: slack=%f, newslack=%f\n", i+1, slack, margin-dist);
      	      }
              */
              if(energy > -sparm->loss_value && (dist + slack) < (margin - epsilon))
              { 
                if(struct_verbosity >= 2)
                {
                  printf("(%i,eps=%.2f) ", i, margin-dist-slack);
                  fflush(stdout);
                }
                if(struct_verbosity == 1)
                {
                  printf(".");
                  fflush(stdout);
                }
	      
	        /**** resize constraint matrix and add new constraint ****/
                cset.m++;
                cset.lhs = (DOC **)realloc(cset.lhs, sizeof(DOC *)*cset.m);
      	        if(kparm->kernel_type == LINEAR)
                {
                  diff = add_list_ss(fy); /* store difference vector directly */
                  if(sparm->slack_norm == 1) 
                    cset.lhs[cset.m-1] = create_example(cset.m-1, 0, init_slack+i+1, 1, copy_svector(diff));
      		  else if(sparm->slack_norm == 2)
                  {
                    /**** add squared slack variable to feature vector ****/
                    slackv[0].wnum = sizePsi + i;
                    slackv[0].weight = 1/(sqrt(2*svmCnorm));
                    slackv[1].wnum = 0; /*terminator*/
                    slackvec = create_svector(slackv, "", 1.0);
                    cset.lhs[cset.m-1] = create_example(cset.m-1, 0, init_slack+i+1, 1, add_ss(diff,slackvec));
       		    free_svector(slackvec);
                  }
                  free_svector(diff);
                }  
                else   /* kernel is used */
                { 
                  if(sparm->slack_norm == 1)
       		    cset.lhs[cset.m-1]=create_example(cset.m-1, 0, init_slack+i+1, 1, copy_svector(fy));
       		  else if(sparm->slack_norm == 2)
                    exit(1);
                }
                cset.rhs = (double *)realloc(cset.rhs, sizeof(double)*cset.m);
                cset.rhs[cset.m-1] = margin;
                alpha = (double *)realloc(alpha,sizeof(double)*cset.m);
                alpha[cset.m-1] = 0;
                alphahist=(long *)realloc(alphahist, sizeof(long)*cset.m);
                alphahist[cset.m-1] = optcount;
                newconstraints++;
                totconstraints++;
              }
              else
              {
                printf("+");
                fflush(stdout); 
                if(opti[i] != opti_round)
                {
                  activenum--;
                  opti[i] = opti_round; 
                }
              }

              free_example(doc, 0);
            }
            free_svector(fy); /* this also free's fybar */
          }
          else if(rank != 0)
          {
            MPI_Recv(svmModel->lin_weights, svmModel->totwords+1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &Stat);
            MPI_Recv(&(svmModel->b), 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &Stat);
            sm->svm_model = svmModel;
            sm->w = svmModel->lin_weights;
            copy_to_float_weights(sm);
          }

          /**** get new QP solution ****/
          if(rank == 0 && (i % procs_num == procs_num-1 || i == n-1))
          {
            if( (newconstraints >= sparm->newconstretrain) || ((newconstraints > 0) && (i == n-1)) || (new_precision && (i == n-1)) )
            {
              if(struct_verbosity >= 1)
              {
                printf("*%ld new constraints ", newconstraints);
                fflush(stdout);
              }
              rt2 = get_runtime();
      	      /* Always get a new kernel cache. It is not possible to use the same cache for two different training runs */
              if(kparm->kernel_type != LINEAR)
                kcache=kernel_cache_init(MAX(cset.m,1), lparm->kernel_cache_size);
              /* Run the QP solver on cset. */
              svm_learn_optimization_linear(cset.lhs, cset.rhs, cset.m, sizePsi+n, init_slack+n, lparm, kparm, kcache, svmModel, alpha, 0);
              if(kcache)
                kernel_cache_cleanup(kcache);
              /* Always add weight vector, in case part of the kernel is linear. If not, ignore the weight vector since its content is bogus. */
              if(kparm->kernel_type != LINEAR)
                add_weight_vector_to_linear_model(svmModel);
              sm->svm_model = svmModel;
              sm->w = svmModel->lin_weights; /* short cut to weight vector */
              copy_to_float_weights(sm);

              optcount++;
              /* keep track of when each constraint was last active. constraints marked with -1 are not updated */
              for(j = 0; j < cset.m; j++) 
              {
                if((alphahist[j] >- 1) && (alpha[j] != 0))  
                  alphahist[j] = optcount;
              }
              rt_opt += MAX(get_runtime()-rt2,0);
	    
              new_precision = 0;
              newconstraints = 0;
            }
            for(j = i; j % procs_num; j--)
            {
              MPI_Send(svmModel->lin_weights, svmModel->totwords+1, MPI_DOUBLE, j % procs_num, 1, MPI_COMM_WORLD);
              MPI_Send(&(svmModel->b), 1, MPI_DOUBLE, j % procs_num, 2, MPI_COMM_WORLD);
            }
          }

          rt_total += MAX(get_runtime()-rt1, 0);

        } /* end of example loop */

        /* broadcast */
        MPI_Bcast(svmModel->lin_weights, svmModel->totwords+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(svmModel->b), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        sm->svm_model = svmModel;
        sm->w = svmModel->lin_weights; /* short cut to weight vector */
        copy_to_float_weights(sm);
        MPI_Bcast(&(activenum), 1, MPI_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(opti, n, MPI_INT, 0, MPI_COMM_WORLD);
        if(rank == 0)
          flag_terminate = ((cset.m - old_numConst) > tolerance) || (!fullround);
        MPI_Bcast(&flag_terminate, 1, MPI_INT, 0, MPI_COMM_WORLD);

        rt1=get_runtime();
	
        if(rank == 0 && struct_verbosity >= 1)
          printf("(NumConst=%d, SV=%ld, CEps=%.4f, QPEps=%.4f)\n", cset.m, svmModel->sv_num-1, ceps, svmModel->maxdiff);
	
      	/* Check if some of the linear constraints have not been active in a while. Those constraints are then removed to avoid bloating the working set beyond necessity. */
        if(rank == 0)
        {
          if(struct_verbosity >= 2)
          {
            printf("Reducing working set...");
            fflush(stdout);
          }
          remove_inactive_constraints(&cset, alpha, optcount, alphahist, MAX(10, optcount-lastoptcount));
          lastoptcount = optcount;
          if(struct_verbosity >= 2)
            printf("done. (NumConst=%d)\n", cset.m);
        }
	
        rt_total += MAX(get_runtime()-rt1, 0);
        MPI_Barrier(MPI_COMM_WORLD);
	
      } while(activenum > 0 && flag_terminate);   /* repeat until all examples produced no constraint at least once */

    } while(0);

  } while(epsilon > sparm->epsilon);  

  if(rank == 0 && struct_verbosity>=1)
  {
    /**** compute sum of slacks ****/
    /**** WARNING: If positivity constraints are used, then the maximum slack id is larger than what is allocated below ****/
    slacks = (double *)my_malloc(sizeof(double)*(init_slack+n+1));
    for(i = 0; i <= init_slack+n; i++)
    { 
      slacks[i] = 0;
    }
    if(sparm->slack_norm == 1)
    {
      for(j = 0; j < cset.m; j++) 
        slacks[cset.lhs[j]->slackid] = MAX(slacks[cset.lhs[j]->slackid], cset.rhs[j]-classify_example(svmModel,cset.lhs[j]));
    }
    else if(sparm->slack_norm == 2)
    {
      for(j = 0; j < cset.m; j++) 
	      slacks[cset.lhs[j]->slackid] = MAX(slacks[cset.lhs[j]->slackid], cset.rhs[j] - (classify_example(svmModel,cset.lhs[j]) - sm->w[sizePsi+cset.lhs[j]->slackid-1]/(sqrt(2*svmCnorm))));
    }
    slacksum = 0;
    for(i = 1; i <= init_slack+n; i++)  
      slacksum += slacks[i];
    free(slacks);

    printf("Total number of constraints in final working set: %i (of %i)\n",(int)cset.m, (int)totconstraints);
    printf("Number of iterations: %d\n", numIt);
    printf("Number of calls to 'find_most_violated_constraint': %ld\n", argmax_count);

    printf("Norm. sum of slack variables (on working set): sum(xi_i)/n=%.5f\n", slacksum/(init_slack+n));
    printf("Norm of longest difference vector: ||Psi(x,y)-Psi(x,ybar)||=%.5f\n", length_of_longest_document_vector(cset.lhs,cset.m,kparm));
    printf("Runtime in cpu-seconds: %.2f (%.2f%% for QP, %.2f%% for Argmax, %.2f%% for Psi, %.2f%% for init)\n", rt_total/100.0, (100.0*rt_opt)/rt_total, (100.0*rt_viol)/rt_total, (100.0*rt_psi)/rt_total, (100.0*rt_init)/rt_total);
  }
  if(rank == 0 && struct_verbosity >= 4)
    printW(sm->w,sizePsi, n, lparm->svm_c);

  if(svmModel) 
  {
    sm->w = sm->svm_model->lin_weights; /* short cut to weight vector */
    copy_to_float_weights(sm);
    sm->svm_model = NULL;
  }

  /* save the precision for future learning */
  sparm->epsilon_init = epsilon;
  if(rank == 0)
    write_constraints(cset, alpha, sparm);

  print_struct_learning_stats(sample, sm, cset, alpha, sparm);

  if(svmModel)
    free_model(svmModel, 0);
  free(opti);
  free(diff_n);
  free(diff_margin_energy);

  if(rank == 0)
  {
    free(alpha); 
    free(alphahist); 
    free(cset.rhs); 
    for(i = 0; i < cset.m; i++) 
      free_example(cset.lhs[i], 1);
    free(cset.lhs);
  }
}


void svm_learn_struct_joint(SAMPLE sample, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm, STRUCTMODEL *sm, int alg_type)
{
  int         i, j;
  int         numIt = 0;
  long        argmax_count = 0;
  long        totconstraints = 0;
  long        kernel_type_org;
  double      epsilon, epsilon_cached;
  double      lossval, factor, dist;
  double      margin = 0;
  double      slack, slacksum, ceps = 0;
  double      dualitygap, modellength, alphasum;
  long        sizePsi;
  double      *alpha = NULL;
  long        *alphahist = NULL, optcount = 0;
  CONSTSET    cset;
  SVECTOR     *diff = NULL;
  double      *diff_n = NULL;
  SVECTOR     *fy, *fybar, *f, *lhs;
  MODEL       *svmModel = NULL;
  LABEL       ybar;
  DOC         *doc;

  long        n = sample.n;
  EXAMPLE     *ex = sample.examples;
  double      rt_total = 0, rt_opt = 0, rt_init = 0, rt_psi = 0, rt_viol = 0, rt_kernel = 0;
  double      rt1, rt2;
  double      progress, progress_old;

  double y_score, ybar_score, azimuth, elevation, distance, prediction, cur_loss;

  CCACHE      *ccache = NULL;
  int         cached_constraint;
  int flag_terminate;

  /* MPI process */
  int rank;
  int procs_num;
  double *diff_margin = NULL, *sum_diff_margin = NULL;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs_num);
  
  rt1 = get_runtime();

  /* initialize model */
  init_struct_model(sample, sm, sparm, lparm, kparm);
  printf("sm->sizePsi = %ld\n", sm->sizePsi);
  sizePsi = sm->sizePsi + 1;          /* sm must contain size of psi on return */

  if(sparm->slack_norm == 1) 
  {
    lparm->svm_c = sparm->C;          /* set upper bound C */
    lparm->sharedslack = 1;
  }
  else if(sparm->slack_norm == 2) 
  {
    printf("ERROR: The joint algorithm does not apply to L2 slack norm!"); 
    fflush(stdout);
    exit(0); 
  }
  else 
  {
    printf("ERROR: Slack norm must be L1 or L2!");
    fflush(stdout);
    exit(0);
  }

  lparm->biased_hyperplane = 0;     /* set threshold to zero */

  /* initialize constraint set */
  if(rank == 0)
  {
    cset = init_struct_constraints(sample, &alpha, sm, sparm);
    if(cset.m > 0) 
    {
      alphahist = (long *)realloc(alphahist,sizeof(long)*cset.m);
      for(i = 0; i < cset.m; i++) 
      {
        alphahist[i] = -1; /* -1 makes sure these constraints are never removed */
      }
    }
    kparm->gram_matrix = NULL;
    if((alg_type == DUAL_ALG) || (alg_type == DUAL_CACHE_ALG))
      kparm->gram_matrix = init_kernel_matrix(&cset, kparm);
  }
  MPI_Bcast(&(sparm->epsilon_init), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  epsilon = sparm->epsilon_init;    /* start with low precision and increase later */
  epsilon_cached = epsilon;         /* epsilon to use for iterations using constraints constructed from the constraint cache */
  printf("Structural SVM learning with initial epsilon %.2f\n", epsilon);
  MPI_Barrier(MPI_COMM_WORLD);

  /* set initial model and slack variables */
  svmModel = (MODEL *)my_malloc(sizeof(MODEL));
  memset(svmModel, 0, sizeof(MODEL));

  if(rank == 0)
  {
    /* Run the QP solver on cset. */
    kernel_type_org = kparm->kernel_type;
    if((alg_type == DUAL_ALG) || (alg_type == DUAL_CACHE_ALG))
      kparm->kernel_type = GRAM; /* use kernel stored in kparm */
    lparm->epsilon_crit = epsilon;
    svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi+n, lparm, kparm, NULL, svmModel, alpha);
    printf("NumConst = %d, SV = %ld, QPEps = %.4f\n", cset.m, svmModel->sv_num-1, svmModel->maxdiff);
    kparm->kernel_type = kernel_type_org; 
    svmModel->kernel_parm.kernel_type = kernel_type_org;
    add_weight_vector_to_linear_model(svmModel);
  }
  MPI_Bcast(&(svmModel->totwords), 1, MPI_LONG, 0, MPI_COMM_WORLD);
  if(rank != 0)
  {
    svmModel->kernel_parm.kernel_type = kparm->kernel_type;
    svmModel->lin_weights = create_nvector(svmModel->totwords);
    clear_nvector(svmModel->lin_weights, svmModel->totwords);
  }
  /* broadcast the weights */
  MPI_Bcast(svmModel->lin_weights, svmModel->totwords+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(svmModel->b), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  sm->svm_model = svmModel;
  sm->w = svmModel->lin_weights; /* short cut to weight vector */
  copy_to_float_weights(sm);

  /* initialize the constraint cache */
  if(alg_type == DUAL_CACHE_ALG) 
    ccache = create_constraint_cache(sample, sparm);

  rt_init += MAX(get_runtime()-rt1, 0);
  rt_total += MAX(get_runtime()-rt1, 0);

  /* select training samples */
  sample = select_examples(sample, sparm->fractions[0]);
  sparm->iter = 0;
  printf("iteration %d/%d with %f training samples\n", sparm->iter+1, ITERATION_NUM, sparm->fractions[sparm->iter]);

  /*** main loop ***/
  do 
  { 
    /* iteratively find and add constraints to working set */
    if(rank == 0 && struct_verbosity >= 1) 
    { 
      printf("Iter %i: ",++numIt); 
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
      
    rt1 = get_runtime();

    /**** compute current slack ****/
    if(rank == 0)
    {
      slack = 0;
      for(j = 0; j < cset.m; j++) 
        slack = MAX(slack, cset.rhs[j]-classify_example(svmModel, cset.lhs[j]));
    }
    MPI_Bcast(&slack, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      
    /**** find a violated joint constraint ****/
    lhs = NULL;
    dist = 0;
    if(alg_type == DUAL_CACHE_ALG) 
    {
      /* see if it is possible to construct violated constraint from cache */
      update_constraint_cache_for_model(ccache, svmModel, rank, procs_num);
      dist = find_most_violated_joint_constraint_in_cache(ccache, &lhs, &margin, rank, procs_num, kparm, sm);
    }

    rt_total += MAX(get_runtime()-rt1, 0);

    /* is there a sufficiently violated constraint in cache? */
    if(dist-slack > MAX(epsilon/10, sparm->epsilon)) 
    { 
      /* use constraint from cache */
      rt1 = get_runtime();
      cached_constraint = 1;
      if(kparm->kernel_type == LINEAR) 
        diff = lhs;
      else 
      { 
        /* Non-linear case: make sure we have deep copy for cset */
	      diff = copy_svector(lhs); 
	      free_svector_shallow(lhs);
      }
      rt_total += MAX(get_runtime()-rt1,0);
    }
    else 
    { 
      /* do not use constraint from cache */
      rt1 = get_runtime();
      cached_constraint = 0;
      if(lhs)
	      free_svector_shallow(lhs);
      lhs = NULL;

      if(kparm->kernel_type == LINEAR) 
      {
	      diff_n = create_nvector(sm->sizePsi);
        clear_nvector(diff_n, sm->sizePsi);
        /* for MPI usage */
	      diff_margin = create_nvector(sm->sizePsi+1);
        sum_diff_margin = create_nvector(sm->sizePsi+1);
      }

      margin = 0;
      progress = 0;
      progress_old = progress;
      rt_total += MAX(get_runtime()-rt1,0);

      /**** find most violated joint constraint ***/
      for(i = 0; i < n; i++)
      {
        if(i % procs_num != rank)
          continue;
	      rt1 = get_runtime();
      
	      progress += 10.0*procs_num / n;
	      if((struct_verbosity==1) && (((int)progress_old) != ((int)progress)))
	      {
          printf(".");
          fflush(stdout);
          progress_old=progress;
        }
	      if(struct_verbosity>=2)
	      {
          printf("."); 
          fflush(stdout);
        }

	      rt2 = get_runtime();
	      argmax_count++;
	      if(sparm->loss_type == SLACK_RESCALING) 
	        ybar=find_most_violated_constraint_slackrescaling(ex[i].x, ex[i].y, sm, sparm);
	      else
	        ybar=find_most_violated_constraint_marginrescaling(ex[i].x, ex[i].y, sm, sparm);
	      rt_viol += MAX(get_runtime()-rt2, 0);
	  
        if(empty_label(ybar)) 
        {
	        printf("Notice: empty label was returned for example (%i)\n", i);
          continue;
	      }

        /* get psi(x,y) and psi(x,ybar) */
        rt2 = get_runtime();
        fy = psi(ex[i].x, ex[i].y, 0, sm, sparm);

        doc = create_example(n, 0, i+1, 1, fy);
        y_score = classify_example(sm->svm_model, doc);
        free_example(doc, 0);

        fybar = psi(ex[i].x, ybar, 1, sm, sparm);

        doc = create_example(n, 0, i+1, 1, fybar);
        ybar_score = classify_example(sm->svm_model, doc);
        free_example(doc, 0);

        if(ybar.object_label == 1)
        {
          azimuth = sm->cads[ybar.cad_label]->objects2d[ybar.view_label]->azimuth;
          elevation = sm->cads[ybar.cad_label]->objects2d[ybar.view_label]->elevation;
          distance = sm->cads[ybar.cad_label]->objects2d[ybar.view_label]->distance;
          printf("example %d, (energy ybar %.2f, view %d, azimuth %.1f, elevation %.1f, distance %.1f) ", i+1, ybar.energy, ybar.view_label, azimuth, elevation, distance); fflush(stdout);
        }
        else
        {
          printf("example %d, ", i+1); fflush(stdout);
        }
        printf("score ybar %.2f, ", ybar_score); fflush(stdout);
        prediction = y_score - ybar_score;
        printf("score %.2f, ", prediction); fflush(stdout);

	      rt_psi += MAX(get_runtime()-rt2, 0);
	      lossval = loss(ex[i].y, ybar, sparm);
	      free_label(ybar);

        printf("loss %.2f, ", lossval); fflush(stdout);

        /* calculate loss */
        cur_loss = lossval - prediction;
        if(cur_loss < 0.0)
          cur_loss = 0.0;
        printf("slack %.2f\n", cur_loss); fflush(stdout);
	  
        /**** scale feature vector and margin by loss ****/
	      if(sparm->loss_type == SLACK_RESCALING)
	        factor = lossval / n;
	      else                     /* do not rescale vector for */
	        factor = 1.0 / n;      /* margin rescaling loss type */

      	for(f = fy; f; f = f->next)
          f->factor *= factor;
	      for(f = fybar; f; f = f->next)
	        f->factor *= -factor;
        append_svector_list(fybar, fy);   /* compute fy-fybar */
	  
	      /**** add current fy-fybar and loss to cache ****/
	      if(alg_type == DUAL_CACHE_ALG) 
        {
	        if(kparm->kernel_type == LINEAR) 
	          add_constraint_to_constraint_cache(ccache, svmModel, i, add_list_ss(fybar), lossval/n, sparm->ccache_size);
	        else
	          add_constraint_to_constraint_cache(ccache, svmModel, i, copy_svector(fybar), lossval/n, sparm->ccache_size);
	      }

        /**** add current fy-fybar to constraint and margin ****/
        if(kparm->kernel_type == LINEAR) 
        {
	        add_list_n_ns(diff_n, fybar, 1.0); /* add fy-fybar to sum */
	        free_svector(fybar);
	      }
	      else 
        {
	        append_svector_list(fybar, lhs);  /* add fy-fybar to vector list */
	        lhs = fybar;
	      }
	      margin += lossval/n;                 /* add loss to rhs */
	  
        rt_total += MAX(get_runtime()-rt1, 0);
      } /* end of example loop */

      rt1 = get_runtime();
      /* create sparse vector from dense sum */
      if(kparm->kernel_type == LINEAR) 
      {
        /* MPI message passing */
        memcpy(diff_margin, diff_n, sizeof(double)*(sm->sizePsi+1));
        diff_margin[sm->sizePsi+1] = margin;
        MPI_Allreduce(diff_margin, sum_diff_margin, sm->sizePsi+2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        memcpy(diff_n, sum_diff_margin, sizeof(double)*(sm->sizePsi+1));
        margin = sum_diff_margin[sm->sizePsi+1];
        free_nvector(diff_margin);
        free_nvector(sum_diff_margin);

        diff = create_svector_n(diff_n, sm->sizePsi, "", 1.0);
	      free_nvector(diff_n);
      }
      else 
        diff = lhs;

      rt_total += MAX(get_runtime()-rt1, 0);
    } /* end of finding most violated joint constraint */

    rt1 = get_runtime();

    if(rank == 0)
    {
      /**** if `error', then add constraint and recompute QP ****/
      doc = create_example(cset.m, 0, 1, 1, diff);
      dist = classify_example(svmModel, doc);
      ceps = MAX(0, margin-dist-slack);
      if(slack > (margin-dist+0.000001)) 
      {
        printf("\nWARNING: Slack of most violated constraint is smaller than slack of working\n");
        printf("         set! There is probably a bug in 'find_most_violated_constraint_*'.\n");
        printf("slack=%f, newslack=%f\n",slack, margin-dist);
      }
      if(ceps > sparm->epsilon) 
      { 
        /**** resize constraint matrix and add new constraint ****/
        cset.lhs = (DOC **)realloc(cset.lhs, sizeof(DOC *)*(cset.m+1));
        if(sparm->slack_norm == 1) 
	        cset.lhs[cset.m] = create_example(cset.m,0,1,1,diff);
        else if(sparm->slack_norm == 2)
	        exit(1);
        cset.rhs = (double *)realloc(cset.rhs, sizeof(double)*(cset.m+1));
        cset.rhs[cset.m] = margin;

        alpha = (double *)realloc(alpha,sizeof(double)*(cset.m+1));
        alpha[cset.m] = 0;
        alphahist = (long *)realloc(alphahist,sizeof(long)*(cset.m+1));
        alphahist[cset.m] = optcount;
        cset.m++;
        totconstraints++;

        if((alg_type == DUAL_ALG) || (alg_type == DUAL_CACHE_ALG)) 
        {
	        if(struct_verbosity>=1) 
          {
	          printf(":");
            fflush(stdout);
          }
          rt2 = get_runtime();
	        kparm->gram_matrix = update_kernel_matrix(kparm->gram_matrix, cset.m-1, &cset, kparm);
	        rt_kernel += MAX(get_runtime()-rt2,0);
	      }
	
        /**** get new QP solution ****/
	      if(struct_verbosity>=1) 
        {
	        printf("*");
          fflush(stdout);
        }
        rt2 = get_runtime();

        /* set svm precision so that higher than eps of most violated constr */
	      if(cached_constraint) 
        {
	        epsilon_cached = MIN(epsilon_cached, MAX(ceps, sparm->epsilon)); 
	        lparm->epsilon_crit = epsilon_cached; 
	      }
	      else 
        {
	        epsilon = MIN(epsilon, MAX(ceps, sparm->epsilon)); /* best eps so far */
	        lparm->epsilon_crit = epsilon; 
	        epsilon_cached = epsilon;
	      }

        /* solve the QP problem only with rank 0 */
        free_model(svmModel, 0);
        svmModel = (MODEL *)my_malloc(sizeof(MODEL));
        /* Run the QP solver on cset. */
	      kernel_type_org = kparm->kernel_type;
	      if((alg_type == DUAL_ALG) || (alg_type == DUAL_CACHE_ALG))
	        kparm->kernel_type = GRAM; /* use kernel stored in kparm */
	      svm_learn_optimization(cset.lhs, cset.rhs, cset.m, sizePsi+n, lparm, kparm, NULL, svmModel, alpha);
	      kparm->kernel_type = kernel_type_org; 
	      svmModel->kernel_parm.kernel_type = kernel_type_org;
	      /* Always add weight vector, in case part of the kernel is linear. If not, ignore the weight vector since its content is bogus. */
	      add_weight_vector_to_linear_model(svmModel);
        
        optcount++;
        /* keep track of when each constraint was last active. constraints marked with -1 are not updated */
	      for(j = 0; j < cset.m; j++) 
        {
	        if((alphahist[j] > -1) && (alpha[j] != 0))  
	          alphahist[j]=optcount;
        }
	      rt_opt += MAX(get_runtime()-rt2, 0);
	
        /* Check if some of the linear constraints have not been active in a while. Those constraints are then removed to avoid bloating the working set beyond necessity. */
      	if(struct_verbosity>=2)
        {
	        printf("Reducing working set...");
          fflush(stdout);
        }
	      remove_inactive_constraints(&cset,alpha,optcount,alphahist,50);
	      if(struct_verbosity>=2)
	        printf("done. (NumConst=%d) ",cset.m);
      }
      else 
	      free_svector(diff);

      free_example(doc, 0);
      flag_terminate = (ceps > sparm->epsilon) || finalize_iteration(ceps, cached_constraint, sample, sm, cset, alpha, sparm);
    } /* end if rank == 0 */
    
    /* broadcast the weights */
    MPI_Bcast(svmModel->lin_weights, svmModel->totwords+1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(svmModel->b), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(flag_terminate), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(epsilon), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(flag_terminate == 0)
    {
      sparm->iter++;
      if(sparm->iter < ITERATION_NUM)
      {
        flag_terminate = 1;
        sample = select_examples(sample, sparm->fractions[sparm->iter]);
        printf("iteration %d/%d with %f training samples\n", sparm->iter+1, ITERATION_NUM, sparm->fractions[sparm->iter]);
      }
    }

    sm->svm_model = svmModel;
    sm->w = svmModel->lin_weights; /* short cut to weight vector */
    copy_to_float_weights(sm);

    if(rank == 0 && struct_verbosity >= 1)
      printf("(NumConst=%d, SV=%ld, CEps=%.4f, QPEps=%.4f)\n",cset.m, svmModel->sv_num-1,ceps,svmModel->maxdiff);
    MPI_Barrier(MPI_COMM_WORLD);
	
    rt_total += MAX(get_runtime()-rt1, 0);

    if(rank != 0)
      free_svector(diff);
    MPI_Barrier(MPI_COMM_WORLD);	
  }
  while(flag_terminate);

  if(rank == 0 && struct_verbosity >= 1) 
  {
    /**** compute sum of slacks ****/
    /**** WARNING: If positivity constraints are used, then the maximum slack id is larger than what is allocated below ****/
    slacksum = 0;
    if(sparm->slack_norm == 1) 
    {
      for(j = 0; j < cset.m; j++) 
	    slacksum = MAX(slacksum, cset.rhs[j]-classify_example(svmModel, cset.lhs[j]));
    }
    else if(sparm->slack_norm == 2)
      exit(1);

    alphasum = 0;
    for(i = 0; i < cset.m; i++)  
      alphasum += alpha[i]*cset.rhs[i];
    modellength = model_length_s(svmModel, kparm);
    dualitygap = (0.5*modellength*modellength+sparm->C*(slacksum+ceps)) - (alphasum-0.5*modellength*modellength);
    
    printf("Final epsilon on KKT-Conditions: %.5f\n", MAX(svmModel->maxdiff, ceps));
    printf("Upper bound on duality gap: %.5f\n", dualitygap);
    printf("Dual objective value: dval=%.5f\n", alphasum-0.5*modellength*modellength);
    printf("Total number of constraints in final working set: %i (of %i)\n", (int)cset.m, (int)totconstraints);
    printf("Number of iterations: %d\n", numIt);
    printf("Number of calls to 'find_most_violated_constraint': %ld\n", argmax_count);
    if(sparm->slack_norm == 1) 
    {
      printf("Number of SV: %ld \n", svmModel->sv_num-1);
      printf("Norm of weight vector: |w|=%.5f\n", model_length_s(svmModel,kparm));
    }
    else if(sparm->slack_norm == 2)
    { 
      printf("Number of SV: %ld (including %ld at upper bound)\n", svmModel->sv_num-1, svmModel->at_upper_bound);
      printf("Norm of weight vector (including L2-loss): |w|=%.5f\n", model_length_s(svmModel,kparm));
    }
    printf("Value of slack variable (on working set): xi=%.5f\n",slacksum);
    printf("Norm of longest difference vector: ||Psi(x,y)-Psi(x,ybar)||=%.5f\n", length_of_longest_document_vector(cset.lhs,cset.m,kparm));
    printf("Runtime in cpu-seconds: %.2f (%.2f%% for QP, %.2f%% for kernel, %.2f%% for Argmax, %.2f%% for Psi, %.2f%% for init)\n",
	   rt_total/100.0, (100.0*rt_opt)/rt_total, (100.0*rt_kernel)/rt_total,
	   (100.0*rt_viol)/rt_total, (100.0*rt_psi)/rt_total, 
	   (100.0*rt_init)/rt_total);

    if(ccache)
    {
      long cnum = 0;
      CCACHEELEM *celem;
      for(i = 0; i < n; i++)
      {
        for(celem = ccache->constlist[i]; celem; celem = celem->next) 
          cnum++;
      }
      printf("Final number of constraints in cache: %ld\n",cnum);
    }
    if(struct_verbosity >= 4)
      printW(sm->w, sizePsi, n, lparm->svm_c);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if(svmModel) 
  {
    sm->svm_model = copy_model(svmModel);
    sm->w = sm->svm_model->lin_weights; /* short cut to weight vector */
    copy_to_float_weights(sm);
  }
  
  /* save the precision for future learning */
  sparm->epsilon_init = epsilon;

  if(rank == 0)
    write_constraints(cset, alpha, sparm);
  MPI_Barrier(MPI_COMM_WORLD);

  if(rank == 0)
    print_struct_learning_stats(sample, sm, cset, alpha, sparm);
  MPI_Barrier(MPI_COMM_WORLD);

  if(ccache)    
    free_constraint_cache(ccache);
  if(svmModel)
    free_model(svmModel, 0);

  if(rank == 0)
  {
    free(alpha); 
    free(alphahist); 
    free(cset.rhs); 
    for(i = 0; i < cset.m; i++) 
      free_example(cset.lhs[i], 1);
    free(cset.lhs);
    if(kparm->gram_matrix)
      free_matrix(kparm->gram_matrix);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}


void remove_inactive_constraints(CONSTSET *cset, double *alpha, 
			         long currentiter, long *alphahist, 
				 long mininactive)
     /* removes the constraints from cset (and alpha) for which
	alphahist indicates that they have not been active for at
	least mininactive iterations */

{  
  long i,m;
  
  m=0;
  for(i=0;i<cset->m;i++) 
  {
    if((alphahist[i]<0) || ((currentiter-alphahist[i]) < mininactive)) 
    {
      /* keep constraints that are marked as -1 or which have recently
         been active */
      cset->lhs[m]=cset->lhs[i];      
      cset->lhs[m]->docnum=m;
      cset->rhs[m]=cset->rhs[i];
      alpha[m]=alpha[i];
      alphahist[m]=alphahist[i];
      m++;
    }
    else 
    {
      free_example(cset->lhs[i],1);
    }
  }
  if(cset->m != m) 
  {
    cset->m=m;
    cset->lhs=(DOC **)realloc(cset->lhs,sizeof(DOC *)*cset->m);
    cset->rhs=(double *)realloc(cset->rhs,sizeof(double)*cset->m);
    /* alpha=realloc(alpha,sizeof(double)*cset->m); */
    /* alphahist=realloc(alphahist,sizeof(long)*cset->m); */
  }
}


MATRIX *init_kernel_matrix(CONSTSET *cset, KERNEL_PARM *kparm) 
     /* assigns a kernelid to each constraint in cset and creates the
	corresponding kernel matrix. */
{
  int i,j;
  CFLOAT kval;
  MATRIX *matrix;

  /* assign kernel id to each new constraint */
  for(i=0;i<cset->m;i++) 
    cset->lhs[i]->kernelid=i;

  /* allocate kernel matrix as necessary */
  matrix=create_matrix(i+50,i+50);

  for(j=0;j<cset->m;j++) 
  {
    for(i=j;i<cset->m;i++) 
    {
      kval=kernel(kparm,cset->lhs[j],cset->lhs[i]);
      matrix->element[j][i]=kval;
      matrix->element[i][j]=kval;
    }
  }
  return(matrix);
}

MATRIX *update_kernel_matrix(MATRIX *matrix, int newpos, CONSTSET *cset, 
			     KERNEL_PARM *kparm) 
     /* assigns new kernelid to constraint in position newpos and
	fills the corresponding part of the kernel matrix */
{
  int i,maxkernelid=0,newid;
  CFLOAT kval;
  double *used;

  /* find free kernelid to assign to new constraint */
  for(i=0;i<cset->m;i++) 
    if(i != newpos)
      maxkernelid=MAX(maxkernelid,cset->lhs[i]->kernelid);
  used=create_nvector(maxkernelid+2);
  clear_nvector(used,maxkernelid+2);
  for(i=0;i<cset->m;i++) 
    if(i != newpos)
      used[cset->lhs[i]->kernelid]=1;
  for(newid=0;used[newid];newid++);
  free_nvector(used);
  cset->lhs[newpos]->kernelid=newid;

  /* extend kernel matrix if necessary */
  maxkernelid=MAX(maxkernelid,newid);
  if((!matrix) || (maxkernelid>=matrix->m))
    matrix=realloc_matrix(matrix,maxkernelid+50,maxkernelid+50);

  for(i=0;i<cset->m;i++) 
  {
    kval=kernel(kparm,cset->lhs[newpos],cset->lhs[i]);
    matrix->element[newid][cset->lhs[i]->kernelid]=kval;
    matrix->element[cset->lhs[i]->kernelid][newid]=kval;
  }
  return(matrix);
}

CCACHE *create_constraint_cache(SAMPLE sample, STRUCT_LEARN_PARM *sparm)
     /* create new constraint cache for training set */
{
  long        n = sample.n;
  EXAMPLE     *ex = sample.examples;
  CCACHE      *ccache;
  int         i;

  ccache = (CCACHE *)malloc(sizeof(CCACHE));
  ccache->n = n;
  ccache->constlist = (CCACHEELEM **)malloc(sizeof(CCACHEELEM *)*n);
  for(i = 0; i < n; i++)  /* add constraint for ybar=y to cache */
  { 
    ccache->constlist[i] = (CCACHEELEM *)malloc(sizeof(CCACHEELEM));
    ccache->constlist[i]->fydelta = create_svector_n(NULL,0,"",1);
    ccache->constlist[i]->rhs = loss(ex[i].y,ex[i].y,sparm)/n;
    ccache->constlist[i]->viol = 0;
    ccache->constlist[i]->next = NULL;
  }

  return(ccache);
}

void free_constraint_cache(CCACHE *ccache)
     /* frees all memory allocated for constraint cache */
{
  CCACHEELEM *celem,*next;
  int i;
  for(i = 0; i < ccache->n; i++) 
  {
    celem = ccache->constlist[i];
    while(celem) 
    {
      free_svector(celem->fydelta);
      next = celem->next;
      free(celem);
      celem = next;
    }
  }
  free(ccache->constlist);
  free(ccache);
}

void add_constraint_to_constraint_cache(CCACHE *ccache, MODEL *svmModel, int exnum, SVECTOR *fydelta, double rhs, int maxconst)
     /* add new constraint fydelta*w>rhs for example exnum to cache,
	if it is more violated than the currently most violated
	constraint in cache. if this grows the number of constraint
	for this example beyond maxconst, then the most unused
	constraint is deleted. the funciton assumes that
	update_constraint_cache_for_model has been run. */
{
  double  viol;
  double  dist_ydelta;
  DOC     *doc_fydelta;
  CCACHEELEM *celem;
  int     cnum;

  doc_fydelta=create_example(1,0,1,1,fydelta);
  dist_ydelta=classify_example(svmModel,doc_fydelta);
  free_example(doc_fydelta,0);  
  viol=rhs-dist_ydelta;

  if(ccache->constlist[exnum] == NULL || (viol-0.000000000001) > ccache->constlist[exnum]->viol) 
  {
    celem=ccache->constlist[exnum];
    ccache->constlist[exnum]=(CCACHEELEM *)malloc(sizeof(CCACHEELEM));
    ccache->constlist[exnum]->next=celem;
    ccache->constlist[exnum]->fydelta=fydelta;
    ccache->constlist[exnum]->rhs=rhs;
    ccache->constlist[exnum]->viol=viol;

    /* remove last constraint in list, if list is longer than maxconst */
    cnum=2;
    for(celem=ccache->constlist[exnum];celem && celem->next && celem->next->next;celem=celem->next)
      cnum++;
    if(cnum>maxconst) 
    {
      free_svector(celem->next->fydelta);
      free(celem->next);
      celem->next=NULL;
    }
  }
  else 
  {
    free_svector(fydelta);
  }
}

void update_constraint_cache_for_model(CCACHE *ccache, MODEL *svmModel, int rank, int procs_num)
     /* update the violation scores according to svmModel and find the
	most violated constraints for each example */
{ 
  int     i;
  double  progress=0,progress_old=0;
  double  maxviol=0;
  double  dist_ydelta;
  DOC     *doc_fydelta;
  CCACHEELEM *celem,*prev,*maxviol_celem,*maxviol_prev;

  for(i = 0; i < ccache->n; i++) 
  { /*** example loop ***/
    if(i % procs_num != rank)
      continue;	  
    progress+=10.0*procs_num/ccache->n;
    if((struct_verbosity==1) && (((int)progress_old) != ((int)progress)))
    {
      printf("+");
      fflush(stdout);
      progress_old=progress;
    }
    if(struct_verbosity>=2)
    {
      printf("+");
      fflush(stdout);
    }
    
    maxviol = 0;
    prev = NULL;
    maxviol_celem = NULL;
    maxviol_prev = NULL;
    for(celem = ccache->constlist[i]; celem; celem = celem->next) 
    {
      doc_fydelta = create_example(1,0,1,1,celem->fydelta);
      dist_ydelta = classify_example(svmModel,doc_fydelta);
      free_example(doc_fydelta,0);
      celem->viol = celem->rhs-dist_ydelta;
      if((celem->viol > maxviol) || (!maxviol_celem)) 
      {
        maxviol=celem->viol;
        maxviol_celem=celem;
        maxviol_prev=prev;
      }
      prev=celem;
    }
    if(maxviol_prev) 
    { /* move max violated constraint to the top of list */
      maxviol_prev->next=maxviol_celem->next;
      maxviol_celem->next=ccache->constlist[i];
      ccache->constlist[i]=maxviol_celem;
    }
  }
}

double find_most_violated_joint_constraint_in_cache(CCACHE *ccache, SVECTOR **lhs, double *margin, int rank, int procs_num, KERNEL_PARM *kparm, STRUCTMODEL *sm)
     /* constructs most violated joint constraint from cache. assumes
	that update_constraint_cache_for_model has been run. NOTE:
	this function returns only a shallow copy of the Psi vectors
	in lhs. So, do not use a deep free, otherwise the case becomes
	invalid. */
{
  double sumviol=0;
  int i;
  SVECTOR *fydelta;
  SVECTOR *diff=NULL;
  double *diff_n = NULL;
  double *diff_margin, *sum_diff_margin;

  (*lhs)=NULL;
  (*margin)=0;

  /**** add all maximally violated fydelta to joint constraint ****/
  for(i = 0; i < ccache->n; i++) 
  { 
    if(i % procs_num != rank)
      continue;
    if(ccache->constlist[i])
    {
      fydelta = copy_svector_shallow(ccache->constlist[i]->fydelta);
      append_svector_list(fydelta, (*lhs));           /* add fydelta to lhs */
      (*lhs) = fydelta;
      (*margin) += ccache->constlist[i]->rhs;   /* add loss to rhs */
      sumviol += ccache->constlist[i]->viol;
    }
  }

  if(kparm->kernel_type == LINEAR) 
  {
    diff_n = create_nvector(sm->sizePsi);
    clear_nvector(diff_n, sm->sizePsi);
    add_list_n_ns(diff_n, *lhs, 1.0); /* add *lhs to sum */

    /* for MPI usage */
    diff_margin = create_nvector(sm->sizePsi+2);
    sum_diff_margin = create_nvector(sm->sizePsi+2);

    /* MPI message passing */
    memcpy(diff_margin, diff_n, sizeof(double)*(sm->sizePsi+1));
    diff_margin[sm->sizePsi+1] = *margin;
    diff_margin[sm->sizePsi+2] = sumviol;
    MPI_Allreduce(diff_margin, sum_diff_margin, sm->sizePsi+3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(diff_n, sum_diff_margin, sizeof(double)*(sm->sizePsi+1));
    *margin = sum_diff_margin[sm->sizePsi+1];
    sumviol = sum_diff_margin[sm->sizePsi+2];
    free_nvector(diff_margin);
    free_nvector(sum_diff_margin);

    diff = create_svector_n(diff_n, sm->sizePsi,"",1.0);
    free_svector_shallow(*lhs);
    *lhs = diff;
    free_nvector(diff_n);
  }

  return(sumviol);
}


/* docs:        Left-hand side of inequalities (x-part) */
/* rhs:         Right-hand side of inequalities */
/* totdoc:      Number of examples in docs/label */
/* totwords:    Number of features (i.e. highest feature index) */
/* n:           Number of slack variables */
/* learn_parm:  Learning paramenters */
/* kernel_parm: Kernel paramenters */
/* kernel_cache:Initialized Cache of size 1*totdoc, if using a kernel. NULL if linear.*/
/* model:       Returns solution as SV expansion (assumed empty before called) */
/* alpha:       Start values for the alpha variables or NULL pointer. The new alpha values are returned after optimization if not NULL. Array must be of size totdoc. */
void svm_learn_optimization_linear(DOC **docs, double *rhs, long int totdoc, long int totwords, long int n, LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm, KERNEL_CACHE *kernel_cache, MODEL *model, double *alpha, int is_init)
{
  int i, j, s, iter, temp, slackid;
  int *index;
  int max_iter = 1000;
	int active_size = totdoc;
  double G, C, d, alpha_old;
  double *QD, *alpha_sum;

  // PG: projected gradient, for shrinking and stopping
  double PG;
  double PGmax_old = PLUS_INFINITY;
  double PGmin_old = MINUS_INFINITY;
  double PGmax_new, PGmin_new;

  double v = 0;
  int nSV = 0;

  if(is_init)
  {
    model->supvec = (DOC **)my_malloc(sizeof(DOC *)*(totdoc+2));
    model->alpha = (double *)my_malloc(sizeof(double)*(totdoc+2));
    model->index = (long *)my_malloc(sizeof(long)*(totdoc+2));
    model->at_upper_bound = 0;
    model->b = 0;	       
    model->supvec[0] = 0;  /* element 0 reserved and empty for now */
    model->alpha[0] = 0;
    model->lin_weights = NULL;
    model->totwords = totwords;
    model->totdoc = totdoc;
    model->kernel_parm = (*kernel_parm);
    model->sv_num = 1;
    model->loo_error = -1;
    model->loo_recall = -1;
    model->loo_precision = -1;
    model->xa_error = -1;
    model->xa_recall = -1;
    model->xa_precision = -1;
    model->maxdiff = 0;

    /* empty constraint set */
    if(alpha == NULL)
    {
      model->lin_weights = create_nvector(totwords);
      clear_nvector(model->lin_weights, totwords);
    }
    else
    {
      model->lin_weights = create_nvector(totwords);	
      clear_nvector(model->lin_weights, totwords);
      for(i = 0; i < totdoc; i++)
        add_vector_ns(model->lin_weights, docs[i]->fvec, alpha[i]);
    }
    return;
  }

  QD = (double*)my_malloc(sizeof(double)*totdoc);
  alpha_sum = (double*)my_malloc(sizeof(double)*n);
  memset(alpha_sum, 0, sizeof(double)*n);
  index = (int*)my_malloc(sizeof(int)*totdoc);
  for(i = 0; i < totdoc; i++)
  {
    QD[i] = sprod_ss(docs[i]->fvec, docs[i]->fvec);
    slackid = docs[i]->slackid - 1;
    alpha_sum[slackid] += alpha[i];
    index[i] = i;
  }

  iter = 0;
  while (iter < max_iter)
  {
    PGmax_new = MINUS_INFINITY;
    PGmin_new = PLUS_INFINITY;

    for (i = 0; i < active_size; i++)
    {
      j = i + rand() % (active_size - i);
      /* swap index[i] and index[j] */
      temp = index[i];
      index[i] = index[j];
      index[j] = temp;
    }

    for (s = 0; s < active_size; s++)
    {
      i = index[s];
      slackid = docs[i]->slackid - 1;

      /* gradient */
      G = classify_example(model, docs[i]) - rhs[i];

      C = learn_parm->svm_c;

      /* projected gradient */
      PG = 0;
      if (alpha[i] == 0)
      {
        if (G > PGmax_old)
	{
	  active_size--;
          temp = index[s];
          index[s] = index[active_size];
          index[active_size] = temp;
	  s--;
	  continue;
	}
	else if (G < 0)
	  PG = G;
      }
      else if (alpha_sum[slackid] == C)
      {
        if (G < PGmin_old)
	{
	  active_size--;
          temp = index[s];
          index[s] = index[active_size];
          index[active_size] = temp;
	  s--;
	  continue;
	}
	else if (G > 0)
	  PG = G;
      }
      else
	PG = G;

      PGmax_new = MAX(PGmax_new, PG);
      PGmin_new = MIN(PGmin_new, PG);

      /* update */
      if(fabs(PG) > 1.0e-12)
      {
        alpha_old = alpha[i];
	alpha[i] = MIN(MAX(alpha[i] - G/QD[i], 0.0), C - alpha_sum[slackid] + alpha[i]);

	d = alpha[i] - alpha_old;
        alpha_sum[slackid] += d;
        add_vector_ns(model->lin_weights, docs[i]->fvec, d);
      }
    }

    iter++;
    if(iter % 10 == 0)
    {
      printf(".");
      fflush(stdout);
    }

    if(PGmax_new - PGmin_new <= learn_parm->epsilon_crit)
    {
      if(active_size == totdoc)
	break;
      else
      {
	active_size = totdoc;
	printf("*");
        fflush(stdout);
	PGmax_old = PLUS_INFINITY;
	PGmin_old = MINUS_INFINITY;
	continue;
      }
    }
    PGmax_old = PGmax_new;
    PGmin_old = PGmin_new;
    if (PGmax_old <= 0)
      PGmax_old = PLUS_INFINITY;
    if (PGmin_old >= 0)
      PGmin_old = MINUS_INFINITY;
  }

  printf("\noptimization finished, #iter = %d\n", iter);
  if (iter >= max_iter)
  printf("\nWARNING: reaching max number of iterations\n\n");

  // calculate objective value
  for(i = 1; i < totwords; i++)
    v += model->lin_weights[i] * model->lin_weights[i];

  for(i = 0; i < totdoc; i++)
  {
    v += alpha[i]*(-2);
    if(alpha[i] > 0)
    ++nSV;
  }
  printf("Objective value = %lf\n",v/2);
  printf("nSV = %d\n",nSV);
  model->sv_num = nSV+1;

  free(QD);
  free(alpha_sum);
  free(index);
}
