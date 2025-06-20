/***********************************************************/
/* COPL : graph to sdp file converter v2.0                 */
/***********************************************************/
/* Make a execution file g2sdp by compiling this file.     */
/***********************************************************/
/* Usage of this program :                                 */
/*                                                         */
/*  ! Format  : g2sdp filename -options                    */
/*                                                         */
/*  @ Option  : 'e' for equal-cut                          */
/*              'b' for box-qp                             */
/*              'm' for maximum-cut                        */
/*              's' for stable set problem                 */
/*                                                         */
/*  # Example : "g2sdp G50 -ebm"  <- This will generate    */
/*               equal-cut,box-qp and max-cut data set.    */       
/*               "g2sdp G50 -mbe" will do the same.        */
/*               and if option is not used, it will make   */
/*               all the possible problems.                */
/***********************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<malloc.h>

/*-- Data structure --*/

typedef struct
{  int nn0;
   int *ja;
   double *an;
} spvec;       

/*-- Functions -- */

int *iAlloc(int  len)
{
  int *r=NULL;
  if (len) 
  { r=(int*)calloc(len,sizeof(int));
    if (!r) printf("Mem alloc fail!"); 
  }
  return r;
} 

void iFree(int **x)
{
  int *r=*x;  
  if (r) 
  { free(r);
    *x=NULL;
  }
} 

double *dAlloc(int  len)
{
  double *r=NULL;  
  if (len) 
  { r=(double*)calloc(len,sizeof(double));
    if (!r)  printf("Mem alloc fail!");                  
  }
  return r;
} 

void dFree(double **x)
{
  double *r=*x;  
  if (r)  
  {
    free(r);
    *x=NULL;
  }
} 

spvec *spvAlloc( int  nnzo )
{
  spvec *r;  
  r=(spvec*)calloc(1,sizeof(spvec));
  if (!r) return 0;
  if (nnzo)
  {
    r->ja=iAlloc(nnzo+2);
    if(!r->ja){ printf("\n Memory1...");return 0; }
    r->an=dAlloc(nnzo+2);
    if(!r->an){ printf("\n Memory2...");return 0; } 
  }
  return r;
}

void spvFree(spvec **a)
{
  spvec *r=*a;  
  if (r) 
  { if(r->ja)
    iFree(&r->ja);
    if(r->an)    
    dFree(&r->an);     
  }   
  free(r);
  *a=NULL;
} 

static double nextf(char** ss)
{
  char*  ptr=*ss;
  double r;
  for (;*ptr==' '&&*ptr!='\n';ptr++);
  for (;*ptr!=' '&&*ptr!='\n';ptr++);
  r=atof(ptr);
  *ss=ptr;
  return r;
} /* nexti */

static int geti(char** ss)
{
  char* ptr=*ss;
  int   r;
  for (; *ptr == ' ' && *ptr !='\n' ; ptr++);
  r=atoi(ptr);
  *ss=ptr;
  return r;
}

static int nexti(char** ss)
{
  char* ptr=*ss;
  int   r;
  for (; *ptr == ' ' && *ptr !='\n' ; ptr++);
  for (; *ptr != ' ' && *ptr !='\n' ; ptr++); 
  r=atoi(ptr);
  *ss=ptr;
  return r;
}
 
void sortSpvec(spvec *row_vec)
{
 int i,j,itmp,non0;
 double ftmp;
 non0 = row_vec->nn0;
 
   for(i=non0-2;i>=0;i--)
   {
     for(j=0;j<=i;j++)
     {
        if(row_vec->ja[j+1]<row_vec->ja[j])
        {
           itmp = row_vec->ja[j+1];
           ftmp = row_vec->an[j+1];
           row_vec->ja[j+1] = row_vec->ja[j];
           row_vec->an[j+1] = row_vec->an[j];
           row_vec->ja[j] = itmp;
           row_vec->an[j] = ftmp;                              
        }
     }     
   }
}



/*--------- Main -------------*/

int main(int argc,char *argv[])
{

FILE *fpt,*fpt_sdp,*fpt_ss,*fpt_SS,*fpt_qp,*fpt_ec,*SSinit;
int nnzo,dim,tmp,row,col,maxrown,count,count1,i,j,fmax=1,
    fss=1,fSS=1,fqp=1,fec=1,flag=0;
double value;
char *lines,*temps;
spvec *row_vec;

lines = calloc(256,1);

/********************************************/
/* Determine whether input properly       ***/
/********************************************/

fpt = fopen(argv[1],"r");

if( (argc>1) && (argc<4) && ( fpt != NULL ) )
printf(" OK right input. Starting data conversion!\n");  
else 
{ printf("There is a problem with the number of arguments.\n");
  printf("You have to use 'exe_name graph_name -ebms'.\n");
  fmax=0;fss=0;fqp=0;fec=0;      
}

if(argc==3)
{
  if( strncmp(argv[2],"-",1) != 0)
  {
    fmax=0;fss=0;fqp=0;fec=0;fSS=0;
    printf("\n You missed '-' sign! Nothing will be done! \n");  
  }
  else
  {  
    fmax=0;fss=0;fqp=0;fec=0;fSS=0;
    flag = strlen(argv[2]);
    if( ( strcspn(argv[2],"e")>0 )&&( strcspn(argv[2],"e")<(unsigned)flag ) ) 
    fec=1;  
    if( ( strcspn(argv[2],"b")>0 )&&( strcspn(argv[2],"b")<(unsigned)flag ) ) 
    fqp=1; 
    if( ( strcspn(argv[2],"m")>0 )&&( strcspn(argv[2],"m")<(unsigned)flag ) ) 
    fmax=1; 
    if( ( strcspn(argv[2],"s")>0 )&&( strcspn(argv[2],"s")<(unsigned)flag ) ) 
    fss=1;
    if( ( strcspn(argv[2],"S")>0 )&&( strcspn(argv[2],"S")<(unsigned)flag ) ) 
    fSS=1;

    if( (fmax!=0) || (fqp!=0) || (fec!=0) || (fss!=0) || (fSS!=0) )
    {
      printf(" Generating :");
      if( fmax==1 ) printf(" Maxcut"); 
      if( fqp==1 ) printf(" Box-qp"); 
      if( fec==1 ) printf(" Equal-cut"); 
      if( fss==1 ) printf(" Stable-set ");
      if( fSS==1 ) printf(" Stable-set2 ");
      printf(" problems. \n");
    } 
    else printf("\n Nothing will be done!");
  }
}
else if(argc==2) printf("Generating All the problems.\n");
else printf("\n Something wrong");

temps = lines;
fgets(temps,200,fpt);
dim = geti(&temps);
nnzo = nexti(&temps);

if(fmax==1)
{
   strcpy(lines,"max");
   strcat(lines,argv[1]);
   strcat(lines,".sdp");	    
   fpt_sdp= fopen(lines, "w" );
   if(!fpt_sdp) printf("Cannot Creat file!!!"); 
}

if(fss==1)
{
   strcpy(lines,"stable");
   strcat(lines,argv[1]);
   strcat(lines,".sdp");	 
   fpt_ss= fopen(lines, "w" );
   if(!fpt_ss) printf("Cannot Creat file!!!"); 
}

if(fSS==1)
{
   strcpy(lines,"SS");
   strcat(lines,argv[1]);
   strcat(lines,".sdp");	 
   fpt_SS= fopen(lines, "w" );
   if(!fpt_SS) printf("Cannot Creat file!!!"); 
}

if(fqp==1)
{
   strcpy(lines,"box");
   strcat(lines,argv[1]);
   strcat(lines,".sdp");    
   fpt_qp= fopen(lines, "w" );
   if(!fpt_qp) printf("Cannot Creat file!!!"); 
}
   
if(fec==1)
{
   strcpy(lines,"ecut");
   strcat(lines,argv[1]);
   strcat(lines,".sdp");	 
   fpt_ec = fopen(lines,"w");
   if(!fpt_ec) printf("\n Cannot creat file %s",lines);
}

   /****************************/
   /* Print ROWS               */
   /****************************/

   if(fmax==1)
   {
     fprintf(fpt_sdp,"ROWS");
     for(i=1;i<=dim;i++)
     { fprintf(fpt_sdp,"\n %s %d %f","E",i,1.0);}
   }

   if(fss==1)
   {
     fprintf(fpt_ss,"ROWS");
     for(i=1;i<=dim+1+nnzo;i++)
     { fprintf(fpt_ss,"\n %s %d %f","E",i,1.0);}
   }

   if(fSS==1)
   {
     fprintf(fpt_SS,"ROWS");
     for(i=1;i<=dim+1+nnzo;i++)
     { fprintf(fpt_SS,"\n %s %d %f","E",i,1.0);}
   }

   if(fqp==1)
   {
     fprintf(fpt_qp,"ROWS");
     for(i=1;i<=dim;i++)
     { fprintf(fpt_qp,"\n %s %d %f","E",i,1.0);}
   }
 
   if(fec==1)
   {
     fprintf(fpt_ec,"ROWS");
     for(i=1;i<=dim+1;i++)
     { fprintf(fpt_ec,"\n %s %d %f","E",i,1.0); }
   
   }

   /**************************/
   /* Print POBJM            */
   /**************************/
   if(fss==1)
   {
     fprintf(fpt_ss,"\nPOBJM");   
     for(i=1;i<=dim;i++) fprintf(fpt_ss,"\n %d %d %f",i,dim+1,-0.5);
     fprintf(fpt_ss,"\nPOBJV");
     fprintf(fpt_ss,"\nCONM");
     for(i=1;i<=dim+1;i++) fprintf(fpt_ss,"\n %d %d %f",i,i,1.0);
   }

   if(fSS==1)
   {
     fprintf(fpt_SS,"\nPOBJM");
     for(i=1;i<=dim;i++) fprintf(fpt_SS,"\n %d %d %f",i,dim+1,-0.5);
     fprintf(fpt_SS,"\nPOBJV");
     fprintf(fpt_SS,"\n 1 %f",4.0*dim);
     fprintf(fpt_SS,"\nCONM");
     for(i=1;i<=dim+1;i++) fprintf(fpt_SS,"\n %d %d %f",i,i,1.0);
   }

   if(fmax==1) fprintf(fpt_sdp,"\nPOBJM");
   if(fqp==1)  fprintf(fpt_qp,"\nPOBJM");
   if(fec==1)  fprintf(fpt_ec,"\nPOBJM");

   /* Do sorting stuff here and return to SET+loc_fp */

   maxrown=1;
   tmp=1; /* indicator for different row */
   count=-1;

   for(i=1;i<=nnzo;i++)
   {
     count++;
     temps = lines;
     fgets(temps,200,fpt);
     row = geti(&temps);
     col = nexti(&temps);
     value = nextf(&temps);
     if(tmp<row) 
     {
       tmp=row; 
       if(maxrown<count) maxrown=count;            
       count=-1;
   } }

   fseek(fpt,0,SEEK_SET);
   temps = lines;
   fgets(temps,200,fpt);
   row_vec=spvAlloc(maxrown);
   row_vec->nn0=maxrown;
   tmp=1;
   count=-1;
   j=0;
   count1=dim+2;
   count1=dim+2;

   do
   {
     temps = lines;
     fgets(temps,20,fpt);
     row = geti(&temps);
     col = nexti(&temps);
     value = nextf(&temps);

     if(tmp<row)
     { 
       if(count > 0) 
       { 
	  row_vec->nn0=count+1;
          sortSpvec(row_vec);
       }
       for(i=0;i<=count;i++)
       {
         if(fmax==1) 
         fprintf(fpt_sdp,"\n %d %d %f",tmp,row_vec->ja[i],row_vec->an[i]);
         if(fqp==1) 
         fprintf(fpt_qp,"\n %d %d %f",tmp,row_vec->ja[i],row_vec->an[i]);      
         if(fec==1) 
         fprintf(fpt_ec,"\n %d %d %f",tmp,row_vec->ja[i],row_vec->an[i]);
   	 if(fss==1)
         { fprintf(fpt_ss,"\n %d %d %f",count1,tmp,1.0);
           fprintf(fpt_ss,"\n %d %d %f",count1,row_vec->ja[i],1.0);
           fprintf(fpt_ss,"\n %d %d %f",count1,dim+1,1.0); 
         }
                          
         if(fSS==1)
         { fprintf(fpt_SS,"\n %d %d %f",count1,tmp,1.0);
           fprintf(fpt_SS,"\n %d %d %f",count1,row_vec->ja[i],1.0);
           fprintf(fpt_SS,"\n %d %d %f",count1,dim+1,1.0); 
         }      
       	 count1++;
       }                     
       count=-1;
       tmp=row;       
     }   
     count++;
     row_vec->ja[count] = col;
     row_vec->an[count] = value;
     j++;
   } while(j<nnzo);

   for(i=0;i<=count;i++)
   {
     if(fmax==1) 
     fprintf(fpt_sdp,"\n %d %d %f",tmp,row_vec->ja[i],row_vec->an[i]);
     if(fqp==1) 
     fprintf(fpt_qp,"\n %d %d %f",tmp,row_vec->ja[i],row_vec->an[i]);
     if(fec==1) 
     fprintf(fpt_ec,"\n %d %d %f",tmp,row_vec->ja[i],row_vec->an[i]);
     if(fss==1)
     { fprintf(fpt_ss,"\n %d %d %f",count1,tmp,1.0);
       fprintf(fpt_ss,"\n %d %d %f",count1,row_vec->ja[i],1.0);
       fprintf(fpt_ss,"\n %d %d %f",count1,dim+1,1.0); 
     }
   }  

   if(fSS==1)
   { fprintf(fpt_SS,"\n %d %d %f",count1,tmp,1.0);
     fprintf(fpt_SS,"\n %d %d %f",count1,row_vec->ja[i],1.0);
     fprintf(fpt_SS,"\n %d %d %f",count1,dim+1,1.0); 
   }
              
   spvFree(&row_vec);
  
   /************************/
   /* Print POBJV          */
   /************************/

   if(fmax==1) fprintf(fpt_sdp,"\nPOBJV");
   if(fec==1) fprintf(fpt_ec,"\nPOBJV");
   if(fqp==1) fprintf(fpt_qp,"\nPOBJV");
   if(fec==1) fprintf(fpt_ec,"\n %d %f",1,-1.0*dim);

   /************************/
   /* Print CONM           */
   /************************/

   if(fmax==1) fprintf(fpt_sdp,"\nCONM");
   if(fqp==1) fprintf(fpt_qp,"\nCONM");
   if(fec==1) fprintf(fpt_ec,"\nCONM");

   if(fmax==1) for(i=1;i<=dim;i++) fprintf(fpt_sdp,"\n %d %d %f",i,i,1.0);
   if(fqp==1)  for(i=1;i<=dim;i++) fprintf(fpt_qp,"\n %d %d %f",i,i,1.0);
   if(fec==1)
   {
     for(i=1;i<=dim;i++) fprintf(fpt_ec,"\n %d %d %f",i,i,1.0);
     for(i=1;i<=dim;i++) fprintf(fpt_ec,"\n %d %d %f",dim+1,i,1.0);
   }

   /************************/
   /* Print CONV           */
   /************************/

   if(fmax==1)  fprintf(fpt_sdp,"\nCONV");
   if(fss==1)   fprintf(fpt_ss,"\nCONV");
   if(fSS==1) 
   {
      fprintf(fpt_SS,"\nCONV");
      for(i=dim+2;i<=dim+nnzo+1;i++) fprintf(fpt_SS,"\n %d 1 %f",i,-1.0);
   }
  
   if(fqp==1)  fprintf(fpt_qp,"\nCONV");
   if(fec==1)
   {  
     fprintf(fpt_ec,"\nCONV");
     fprintf(fpt_ec,"\n %d %d %f",dim+1,1,-1.0);
   }

   if(fqp==1) for(i=1;i<=dim;i++) fprintf(fpt_qp,"\n %d %d %f",i,i,1.0);

   /************************/
   /* Print ENDATA         */
   /************************/

   if(fmax==1) fprintf(fpt_sdp,"\nENDATA \n");
   if(fss==1) fprintf(fpt_ss,"\nENDATA \n");
   if(fqp==1) fprintf(fpt_qp,"\nENDATA \n");
   if(fec==1) fprintf(fpt_ec,"\nENDATA \n");
   if(fSS==1) fprintf(fpt_SS,"\nENDATA \n");
   free(lines);
   if(fmax==1) fclose(fpt_sdp);
   if(fss==1) fclose(fpt_ss);
   if(fqp==1) fclose(fpt_qp);
   if(fec==1) fclose(fpt_ec);
   if(fSS==1) fclose(fpt_SS);
   fclose(fpt);
   return 0;
}
