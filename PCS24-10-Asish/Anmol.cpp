#include<bits/stdc++.h>
using namespace std;


void my_func(string str,int k){

    int slow=0;
    int fast=0;
    int n=str.length();
    int cc=0;
    int dd=0;

    vector<string>num;

    

    while(slow<n && fast<n){

        string temp="";

        int arr[26]={0};
        int flag=0;

        cout<<endl;
        while(dd<k && fast<n){

    
            if(arr[str[fast]-'a']==0){
                arr[str[fast]-'a']++;
                temp=temp+str[fast];
                dd++;
                fast++;
                flag=0;
            }
            else{
                arr[str[fast]-'a']++;
                temp=temp+str[fast];
                fast++;
            }

            if(dd==k){

                cc++;
                int flag=0;

                while(arr[str[fast]-'a']!=0){
                    cc++;
                    fast++;
                    arr[str[fast]-'a']++;
                    temp=temp+str[fast];
                }


                
                slow++;
                dd=0;
                fast=slow;
                num.push_back(temp);
                break;
            }
        }
    }

    for(auto i:num){
        cout<<i<<" "<<endl;;
    }

    cout<<cc;
    //return cc;
}

int main(){

    string str="aacfssa";
    int k=3;

    my_func(str,k);
}