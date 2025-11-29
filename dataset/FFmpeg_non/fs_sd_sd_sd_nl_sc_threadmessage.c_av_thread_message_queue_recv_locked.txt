static int func1 ( AVThreadMessageQueue * var1 , 
void * var2 , 
unsigned var3 ) 
{ 
while ( ! var1 -> var4 && func2 ( var1 -> var5 ) < var1 -> var6 ) { 
if ( ( var3 & var7 ) ) 
return func3 ( var8 ) ; 
func4 ( & var1 -> var9 , & var1 -> var10 ) ; 
} 
if ( func2 ( var1 -> var5 ) < var1 -> var6 ) 
return var1 -> var4 ; 
func5 ( var1 -> var5 , var2 , var1 -> var6 , var11 ) ; 
func6 ( & var1 -> var9 ) ; 
return number ; 
} 
