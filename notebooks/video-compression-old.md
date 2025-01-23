```python
!pip install datasets
!pip install numba
```

    Requirement already satisfied: datasets in /usr/local/lib/python3.10/dist-packages (2.19.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from datasets) (3.14.0)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from datasets) (1.25.2)
    Requirement already satisfied: pyarrow>=12.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (14.0.2)
    Requirement already satisfied: pyarrow-hotfix in /usr/local/lib/python3.10/dist-packages (from datasets) (0.6)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.3.8)
    Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (2.0.3)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (2.31.0)
    Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (4.66.4)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets) (3.4.1)
    Requirement already satisfied: multiprocess in /usr/local/lib/python3.10/dist-packages (from datasets) (0.70.16)
    Requirement already satisfied: fsspec[http]<=2024.3.1,>=2023.1.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (2023.6.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.9.5)
    Requirement already satisfied: huggingface-hub>=0.21.2 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.23.1)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from datasets) (24.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (6.0.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (23.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.9.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.21.2->datasets) (4.11.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (2024.2.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2023.4)
    Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.16.0)
    Requirement already satisfied: numba in /usr/local/lib/python3.10/dist-packages (0.58.1)
    Requirement already satisfied: llvmlite<0.42,>=0.41.0dev0 in /usr/local/lib/python3.10/dist-packages (from numba) (0.41.1)
    Requirement already satisfied: numpy<1.27,>=1.22 in /usr/local/lib/python3.10/dist-packages (from numba) (1.25.2)



```python
from pathlib import Path
```


```python
list(Path('/').glob('./root/**/778a08cdcd6718b73373d7d5fa7c3104_9.npy'))
```




    []




```python
# next(iter(ds))
```


```python
# tokens = np.load('~/.cache/huggingface/datasets/downloads/' + ds['0'][0]['path']) # first segment from the first data shard
```


```python
ds['11']['path']
```




    ['e0dd39cccca0e6ffb7d8c7da4797c244_169.npy',
     '59f353f4a2c072119360e4949be7a4f1_14.npy',
     '5765d589395b22c88a90e96ee29150ec_3.npy',
     'ca8bcb97c4a327351262dae1d577cfd2_28.npy',
     'dadd591d0dacc025d002047f83803957_18.npy',
     'c3bc08f693e9b186a839a95c826ac5ae_1.npy',
     '741842915bb6091945a13c5d5abd79bf_2.npy',
     '04de7c5028d9f258af265397414993b4_10.npy',
     '4e24a6a1e4c6e02a44e18d54cfbfa2b0_25.npy',
     'c6a64f69e7b58a9e8dfbdc299f57e3a3_30.npy',
     'c2b409d0b4ab9a5075420909e4ca2c50_5.npy',
     '17760cee0317b65fc01d5df0a013073f_8.npy',
     '34a4e8b4b77351fb06f292643e348aaa_9.npy',
     '9a36305ac3adf9847076943136bf8ab8_76.npy',
     'b98256b14fd54dce90e547de5f550436_104.npy',
     '9706ede4d0222364a9e479107df0910c_4.npy',
     'ee709bc9e1df626c508ebb371731f237_18.npy',
     'bb20d0eecece146af0634c4745881b22_2.npy',
     '97089324137a2fe63a39fc665565da82_45.npy',
     '6079ed7613a5fc58d593ccacc0b05c04_8.npy',
     '25bfdf102ef1611f6746bee91fe54469_1.npy',
     '3d2073e45fd34d9dbe70786f92f7a558_14.npy',
     '40ee628ac14ed1db20845fb3cec133bd_2.npy',
     'ecde2a93a9cec1aad27bcc43669d1df4_81.npy',
     '090bb37e2f8b9c637cae047c7cc91579_202.npy',
     '6f2a9ed451ec4757ab4492d9dfe7a17f_91.npy',
     '8e5ba0f5632c01e93b245ff33e9b1e6a_4.npy',
     '5669c8ea5ffec1dd58e94ccf7578c80e_19.npy',
     '179b5cd900d47f76beb8c0543364d8e2_48.npy',
     '8fb4b3e24ee411e31eca91c54390a628_31.npy',
     '95b1c91092b893226996e9b2e247d1f8_10.npy',
     '742fb3e3f19343e7f14e48386bd50b6b_14.npy',
     '299d1273f3a9c732cc1b320833f6074e_4.npy',
     '54f96aa90f8a483731e7c69987833bbc_12.npy',
     'b7ca51f56fe703bf52402d673cbf07a0_29.npy',
     '205fd52e7549638dd5ea71dd1040c1fa_0.npy',
     'b67014d88fed51d96658168afc471d92_3.npy',
     '1187756e1776753d188211bbc30c7014_79.npy',
     '37d37b78c3fa0ac3aa7364c5122946a2_25.npy',
     '705e3c711b7a3b6854e8d2c8d2c5c1a3_49.npy',
     'f6800dc6d9b5f9826c2b4ebc538843e5_12.npy',
     '0f22fb58a6e8576962a99145e1a23461_20.npy',
     '1796d1f161298a2815b8c25ddc29cb46_1.npy',
     '2a59bcf6ac288647fe65d69cbecab18c_12.npy',
     'a4ce8c522f5b1e2cd4b049aabc9c708f_7.npy',
     '26f6509dad8effcc9000163c05020844_9.npy',
     'b46b2cb87c88fe6c494a074f4fc1f375_40.npy',
     'dd2b63e5d6e8c6ccf9da737ec13e54f6_40.npy',
     '0ed18c4467f1ce13118bb95a35894e51_80.npy',
     'c26327e836f8b6d230dddc0b43e66853_14.npy',
     '3a61a81256ae203870468dae3caca917_5.npy',
     '8a2329775e69da743230f1c1dc6639c0_5.npy',
     '2f7d05cca95ef832a3e52624a6a20904_2.npy',
     '30a718e8d16c23a013cd280e6953cc10_8.npy',
     'a38dbfe513b846807e750776f8e3493c_30.npy',
     'b067dc3547930adf6325498c4c8ebf57_29.npy',
     '5b97c28ba4afb377e06e86bf61606047_6.npy',
     'c72436b39b0c3ad2e09e9fedb2b9864e_6.npy',
     '7ed619fa4bfb39822ae41d6451fbd1d6_178.npy',
     '00dde7507d88c13035d1655f8dd6f8f8_1.npy',
     'a56f56caa5d46d54d4e8690d8fb0bbc7_2.npy',
     '823435fda18d81952781e1ddef34be89_12.npy',
     'c42661b02ef467813fdb191a6e42ef24_9.npy',
     '94f5a7416711ae45d90fb76145756281_3.npy',
     'b3629e01351435f7c80179217cb74428_299.npy',
     'e0189c329b61bf059997a8602ad8d332_8.npy',
     '6c225c3737dac59b0d7febe66e1ad55c_27.npy',
     '1b59c90619d521ab70eee0ac66226832_8.npy',
     'dc603cd5dd08302d93233ad358382049_1.npy',
     '927b29f661a514289e220b6b67a15d21_37.npy',
     '4a2796df6fc11bcfb785957e196eeed5_3.npy',
     '27e379e6c6e0e081fb9ce2748907c815_7.npy',
     '0a676cea339cbc5b87c3bf5444ef6dc6_19.npy',
     'd5574bca722bdc434673ada5e684d4a4_55.npy',
     'e4866d8555441a419a9de6f3b2c58a9d_9.npy',
     '143cf2501f58c7b3e4bcc32efe95022d_6.npy',
     '9ce597aa6d0b493ac316b54b17e837f7_21.npy',
     '1d02e1de263f82c4fb34d91bc4dca1da_4.npy',
     '5322a7599481b600a00697bf599f724b_17.npy',
     '3a3ad7459702162650595d96ccfa5e58_13.npy',
     '7a8ddb01004a10c0d02d15f64821eb07_396.npy',
     'a41935af237a068d643bc052a80f354d_6.npy',
     '3044c097d6c411e701c03a3cd15b18cf_1.npy',
     '48db22b86cbd53fc4ae18089fdde639d_10.npy',
     '64324da6e14411f42ce092448d34ec78_6.npy',
     '8042296cb87a39fa235693d3a1b5f54b_9.npy',
     'a6f6d49594c535ba7fb84b0649166d12_6.npy',
     '89daadc2982e64c5837faa17d1632c2e_0.npy',
     'dbb65ab8362d777676829468fb111ec2_6.npy',
     'eff6f5e1c9403cd656e502e30f785aa5_7.npy',
     'e12d8d0310ba53d3fcc3791a449bc239_14.npy',
     '1b41c2af5d903060457faedc65b2313d_15.npy',
     '6fe9bfed8c168ff15a732587f7d945b5_108.npy',
     '1d1c45513f590e7344e78d7c817b8b71_4.npy',
     '69dbc32053a08e09a72075127f2e56fe_4.npy',
     'a2c6d312041dafe9022674947aa61a14_64.npy',
     'c7b2a32b4512000e8148141acfbb6e26_14.npy',
     '3f27d22c6e66d36b565ee10572f94c55_3.npy',
     '46bbf3b68892600a9ad42438bf723bfa_21.npy',
     'fe379c8fb8302cf89477c1455e293daf_29.npy',
     'a940983ee6461c8d3b168c0965ff5581_20.npy',
     'c76ca6df7498dafcb92e5d2a1c7bb0e7_30.npy',
     '3c51328e7204960515a7553f05ac4002_7.npy',
     'c927eea32f28f8bb6e7158ffe786607d_4.npy',
     'efe830ea9aaf92c052f04a40cb52e641_4.npy',
     'd1dd8f686ca37cea4ea810782544a119_16.npy',
     '077189249c94bd8b49f8e1d975488c67_21.npy',
     '946890986d6edb39124634613e84aa7d_5.npy',
     '5bdda995495187aa4483b6acf5150bde_9.npy',
     '4dee705c166bd7610ffbc265233c4339_17.npy',
     '90ad0dff3d43245d122a9016dcd0327f_44.npy',
     '63152b836b99d32e1a4290e26f27535f_96.npy',
     'c306b94976b25f767897d07bd22a383b_34.npy',
     'b3d673777e3243b7938321951e1cabd4_10.npy',
     '6097eda4dac0725fc535502e925c3c08_13.npy',
     '261c2d9df1e2c1adc88d82d55b25139f_5.npy',
     'f14dcbe3f67114855559c221fd2844c2_56.npy',
     '6a60fe58cdb0283e27b76ad79600e0df_31.npy',
     '3e5a8f6918133be24c0567b54966cb0f_37.npy',
     '66d612d58d38b780b1b03a4f92cc184f_6.npy',
     'f821a38c62c5ebcd255b3798e10818c8_30.npy',
     'fcd196903a8f745782ada2fefc8d070c_8.npy',
     '6e9d7c9db9ac0a9077771b70dca0bc9a_38.npy',
     '3736b56e1db2f8068ed11a6b746f4e1d_28.npy',
     'fca6b706dc8be60a8de4c827cdfea227_20.npy',
     'd19b02ed7dd34c8a5a13050376d3ed08_13.npy',
     'be4a18294adb8ca32e988e0dd9f1f66e_3.npy',
     '35e9a50e0d0aa9add82fe1b77b109f23_20.npy',
     '7cbeae4ed607fb8c61c2cfd0483bbbca_4.npy',
     '1f047041f5ed33058715552493700de6_42.npy',
     '153395ba3b8fab7bcf67a885e56ffb79_95.npy',
     'e070adef81514e39eb64e6c7cb21bce1_1.npy',
     '5dc0479e2d7581dd394ac6649b64e5cc_75.npy',
     '8a8d9bf884d6a321d433ded72c746f4f_15.npy',
     '74032fb8e0ca017a401f28d8082b8b43_5.npy',
     'e4b041921f176c43f3d5163a5c766c0a_32.npy',
     'c4f67c6cce3831673898bd855960e074_13.npy',
     '77947408f171821d69258b832386fd90_13.npy',
     '5c3652db3bcc723b38373b219bda3092_14.npy',
     '9396392559f62639bdeecfa84121d78d_31.npy',
     '0bb5da8379943f2529169c8f2d8da1a0_15.npy',
     '191a73ba3ee0919e0f6654ec1fc004d9_2.npy',
     'b9ee57c0a909c54ccce1358d82c2cce4_22.npy',
     '7818e7b4205a3e38a11953e5a17d20a6_9.npy',
     '906b78f9e1b2bb1ca4e26100d2c7aa9b_20.npy',
     'bfa427e64a773e4e43102000fafec01f_7.npy',
     '3d864e4b16486293bb31fe10247ecffb_3.npy',
     'c8c7a0956bf78ffc4517991a8ebb0c0c_94.npy',
     'da442e37bd0162d8e89a5212b6909d80_28.npy',
     'ebb893cdc2e0ad60c960ab11a755053f_10.npy',
     '16ad78df491abd6de96fca9f205ea64b_2.npy',
     '8790f097d4c92f01b5607292608fd48f_21.npy',
     'cc4d8ec4f009083e4ce6c02c546e3989_7.npy',
     '620a2e1c12778f184130ea0aa4485975_1.npy',
     'e7a941a82fd339728d336ec8bd991628_35.npy',
     '971d5deece6dd292ecac20dcd2390407_6.npy',
     '06800f80958d27c8e84877a698b5037c_7.npy',
     '01756d9b149571919ebcd67b2a66d15c_49.npy',
     '6f56f11a95d3d25a2ea8a2e4e9fb6e6d_9.npy',
     '23d7eae2b69284d34c7e00b6d080931c_3.npy',
     'd15e232794dff911e8f420194a5d7d9a_2.npy',
     '4b68dfb242b9c7f29763f559b91f6167_8.npy',
     '0843a1b933bb7c9c5c996c8fe7dac239_33.npy',
     '4d6da5537bb5f17b53c02944006fbd5d_16.npy',
     '401f87e27f3ef5c3862995c5426e09f4_32.npy',
     'a37226004f914ff1e789baeca79b48ec_29.npy',
     'b9b6b6053246d0b5fdff6ad5bc0f3934_4.npy',
     '5f73f4c6e8fb4504e944ec7a4df27f66_13.npy',
     '2ac657621bac943472a256895475ac23_42.npy',
     'c6d3a1c65394bf7cbd3776bf00f72893_4.npy',
     'bbe7f156cb6198c0331e0b721bd0b115_7.npy',
     '1c7a7fdc7dcac0af208872772b23bca2_6.npy',
     '0d7553938c755bffbd6f4ad71a792f73_6.npy',
     '26d671d4fa396923292457d2bce112e7_9.npy',
     'dc08b290c042716c31bb9172d822866f_5.npy',
     '7b17c126199fea16de275df79615cea4_13.npy',
     '8d8f6aa8dc00424f1a44737a2d1b5722_4.npy',
     'aa978a9fe8bdb77cb68dd5367cbbb355_3.npy',
     'df1244e2b9257e77ea4c2632609b4a32_7.npy',
     'ec061652bd4200241820473223de9dc9_47.npy',
     'e7bd93feb7dec0dd46ae641392d5de32_29.npy',
     '3098c46b9a8af51ca0dd42c46c96731a_8.npy',
     '514520784bd7b5e0cfacddce40092428_12.npy',
     '2f7d05cca95ef832a3e52624a6a20904_23.npy',
     'e69b1f76179b3ee1078e589152f1fbf2_18.npy',
     '8b1572f8e786f8e6100ff9955eea084e_32.npy',
     'd4cc2776251a7387f3c76b562ff54d01_14.npy',
     '6997082314f2e924e68e70cb941b878e_25.npy',
     '3f803ac0c802b633f40afc1a76d7b73c_8.npy',
     '41359a6a810d9134dc03ea68582ab060_34.npy',
     '26e564025a0adc224801723c5916f3f6_38.npy',
     'c4d130693485f22decf928f7e7c0b675_13.npy',
     '825bcc555e8a3ec8c3b81364ae4c6918_3.npy',
     '522ab4893a35633c5f76003e89e8f03d_10.npy',
     'c0b1c361e16895594fa8a94b46f657af_19.npy',
     '62d968e64996022b7c4549ab52552a4f_92.npy',
     'd8023efbd5ed7a46dd955a887f468bec_61.npy',
     '61f0911ea2f0977117281a42ad999472_4.npy',
     'a28bfd3ccdd84db159c77f088497d0c8_10.npy',
     'b485eb4299d9e06985d7a06d6ed04d38_19.npy',
     'b13019f3f37cbbe1a2a8dc913637b8e7_4.npy',
     '3e7e129ea91ecdcbb19c83c21e2f78d8_36.npy',
     '34c6d71f17e15029953333bf45173e57_8.npy',
     '87ff2b78d66f39e502fac59a84f03633_56.npy',
     '541928bf439eb2bb044b2ba8758f39eb_5.npy',
     '540c8d602e10da0fb209e977191fb640_16.npy',
     '3990eab0688be208c9e8fe3c0bd64445_19.npy',
     '7bc1b3ede09efd0113d3e66843d65832_33.npy',
     '07bbb3d9e095837cb26012827154e8c1_9.npy',
     '1f2b0d42b292599e245907d9ac6336cf_4.npy',
     '73b44caab43002ccc83dfd26f04be005_61.npy',
     '059bc89edc8f6e6a611ea7280bf0b8d6_10.npy',
     'a3bf8a568a18fb69db226b56a57fa479_77.npy',
     '438ebf5ffcdd1c2e816cd9bd9bbe1315_6.npy',
     '2b24fea4c804801f3ed1a11a17a887b7_2.npy',
     'c749b41faa5e389018a82a2ed6a4d401_8.npy',
     'b1660ee0937284ee665857d2cdbb97e6_18.npy',
     '150c7fb3e58146bb9ca07e87f02444db_6.npy',
     'f612be8d0aa58bf2adea31a9be89b43d_6.npy',
     '64f83f6d95303d2bb449ea8d9e5301cb_76.npy',
     'b47fd5c5c01660bbec4adcebe41d1e16_54.npy',
     '1edf73077592004f53550007f0280977_2.npy',
     'b5497a1e715a499346f7d0c02834495a_8.npy',
     'c79683050b0087e8b05726d0e9025a2f_6.npy',
     'ba9d95c746f12c60322d57f198f42541_35.npy',
     '9dd7deb9fcd968542b601bde6a2cd70d_7.npy',
     '9d3c13a8d467c5e127dabd135c029bae_5.npy',
     '5a29f131d846c0229cfd15bbce128558_6.npy',
     '3506c737f96390894971b5f14a548f28_10.npy',
     '2b700e5b738446897212dad90d341f10_2.npy',
     '01ca939d1fe8634a06e4a49cca6bd4a3_130.npy',
     'b2676c499b6c44973469d33e56480882_7.npy',
     'f17ed5bdb3a4ee5f16b97f2160244a03_12.npy',
     '79e947d25c0fcfc1eac929a5f21e0843_22.npy',
     'd4c2975e3e8ed43a4062c677cc3cf9a2_48.npy',
     '3ab0dd76683de346be56c12a9254c5c7_29.npy',
     'ba2ce91c94adc8e9e675fc9346f940c2_16.npy',
     'ca2a0cf428e94a255942c5e12a2df0ae_10.npy',
     'c8a0b8741d0494edb62a90ea3437a5b2_3.npy',
     '9a061770dc9a787d175a124ef068c96c_2.npy',
     '65e4159b3e9cdce9dd96c89baa9ae0a7_29.npy',
     'd038b3e8b8f4a4228359be737122110f_5.npy',
     '843a9866878dca98ce631cfd3c7f2e22_78.npy',
     'dcfb10da7f5ba822ffbe38be2568ac8b_20.npy',
     '1e0e78eb29752c017b1416aaeafc45a7_28.npy',
     '7a2eb5531f4d14aad40a530ae31f55f8_21.npy',
     '810d05d1fe47ab42be63e271609d441f_14.npy',
     '3f4161054e0ff9ceaa6b0914cba115f5_7.npy',
     '456f46f096251d326607b63a9fb0f4b7_32.npy',
     '348aec03fe8b96fbe1c9cc25c6a3ce6d_6.npy',
     '908b6ca5ff4c9c8c564e1420b4c699a1_42.npy',
     '3f8c3be0762896df56fc4a42302c3416_51.npy',
     '00c87de2637cfd4b86d1075e92611df5_9.npy',
     'b4a26df8a6f1c2605df45fb7c4024691_3.npy',
     '516e505b8b0beba0adf86d54e9d76668_1.npy',
     '71ca7151d86b4bb21440b3d5cef9b587_9.npy',
     '13e2eb4676db39332c0814984ca06f95_18.npy',
     '85fb394ab1347f94c1b9261894565ede_26.npy',
     'eff81835cdb08d08e88c928e4fdbc989_20.npy',
     '33b1478e03f87e9eb820a2113dc060b0_6.npy',
     '67f251f2c3c1e9449d10c83a92530628_42.npy',
     'ed0321b0b1bffc92c4558d64acb7a5a4_2.npy',
     '7d93dbd6565e168da02ad91183111319_14.npy',
     '101d4c47947fb35932877b123a0c3026_25.npy',
     '7a6889c2954ccc641f8596bd017e7edd_1.npy',
     'e3f4ecad7ea5966fed35d6f003ef852b_9.npy',
     'a92a5e45b637297720a3df3fbe109297_29.npy',
     '3187f575c5cf9db720a2bff27e1a424b_62.npy',
     'e14e47e7e9aa1cf646e6d0ab3843cb7a_7.npy',
     '2092f54ca0ee8eeba900ebb0defc972d_161.npy',
     'da6df533bdba21b606861362df27e395_3.npy',
     '56a3bf6e31b48f86eb1abad074390d1e_2.npy',
     'fae0225a1c550c8a9ecd0e89f7766fbe_14.npy',
     '79c14fca9baff3a956011ca2a75dfdcd_8.npy',
     '0b23eb8cd7ec142900cbfb5e381e6790_18.npy',
     '670e86780cbd7c7ae228d8255462359a_41.npy',
     '69fa655bd9f4bdf9f504a4cb6d8f411d_6.npy',
     '1aec614147f752054b29a0fa5d83e8e5_310.npy',
     '7624dd281b5a04542dbc00696e823c78_9.npy',
     '4df0e106e42dc6cdbb937706ee6a8f67_12.npy',
     '7cace1915d562797739ff68c862f81ac_2.npy',
     'df58dbadb45a3b7fb9e475ce98099324_247.npy',
     'b6f1bb10bd2166cdcfaec1a4c9f45c03_183.npy',
     'a5fe409774594da4a3b082ad3c8ee863_39.npy',
     'f5c716758df7a1e2d11c25c5d66cc7ab_19.npy',
     '3999201ce2c9b21754cfa86a830e67d0_9.npy',
     '3668119f2e0d6b3218060943a697d08a_15.npy',
     'e9ddc2c69495e4ac3d28786897128d03_9.npy',
     '9dfaae3540e6b1859d34bcc81907be22_11.npy',
     '1234409ead41b27f3ff6f51c78351b2d_1.npy',
     'aa8e373f71fe691bf909f60b972d3fe7_16.npy',
     '034a1d3945fd32f561b998e6bd3f9d09_10.npy',
     '9235784274932e79e600988a709f90ad_49.npy',
     '6a5f4aca6e39ec67dbb0a191d3534ba2_4.npy',
     '23d8f904de51d5a7393799c35a5f666c_20.npy',
     '3f5dabd7d75d2c86873edd9e33fa53bb_1.npy',
     'ad3898eecad1e4feef319e50729bc596_1.npy',
     '770c28123b0c919cc19e53a17f00134d_46.npy',
     'e565a0cadb5da7ce400b55c0b5505648_44.npy',
     'dd74e329166ac444f402a8e2fec84384_31.npy',
     '2ffe03b368419602d3186b1b9bc9125b_24.npy',
     '60c5b3297f0e105ce8bd14bf01caef3a_6.npy',
     '6056e65bb18219cdf808ec518a0801aa_41.npy',
     '84f7ddc12596263a28536292962722d3_8.npy',
     '8fc34991ec3f8ef8be73fd290357c334_23.npy',
     '1ca16de0037dc39df6d27fbbbdc2b9fc_8.npy',
     '45eb72f5ab589ee9099eedd5abc5531c_7.npy',
     '53618507b8cfc8c56d63631bb0ed2d78_24.npy',
     'f06e5cdaaf9d897d73346cf36e8a0701_16.npy',
     'a133241fd1697d4e6cd39a6ebaa94a75_16.npy',
     'fae60c70bf891dce7269ef5240ad0926_3.npy',
     'abe8c42c49c54c2df90df005d7aeeefd_23.npy',
     'a6b3f161ceda10664c196d21fa12431f_184.npy',
     'e8e0f57895847e478b64a81cf0863170_7.npy',
     'ec22b40dae344f43b047f4e05f52a4e2_9.npy',
     '9bbd365ad9906871e9eac8507e744053_4.npy',
     '125bb41d7ccca7cf5fa24550d352eb54_10.npy',
     '04a16f33bd0aec86bc6c04c04e0cc44f_21.npy',
     '3ef6dd32c891bb44990d1725eadf397c_11.npy',
     '378ee506ab64081051937149203b7a81_7.npy',
     '0e87f238fac6501be94eaa678852709b_25.npy',
     'ca44b5c85499615baa3a3d30bc3d0646_42.npy',
     'ffb1035fe1454ab3ebfccad8124c327f_26.npy',
     '1d091fed1404753d33efe10b4a058f7c_23.npy',
     '3b251550edb08303e5268ef08ad3e01d_22.npy',
     'c6613bb2259019d7a0189200a40a6c4e_9.npy',
     '77bad15b1ba722dac6d96b48b72d78e3_4.npy',
     '70d802716f68ed5ff0a0e6655b3ca539_6.npy',
     'b3629e01351435f7c80179217cb74428_310.npy',
     'b815d05ead9931823445e6f55af8e346_7.npy',
     'feb7ff8ef24b7765a6594aa98bc5352c_21.npy',
     '89048c43e6c03ad16b7c758d6f4b89ce_16.npy',
     '61413296b0c22e89d7c22ff8827bd3e7_1.npy',
     '8450077505794db6db85a7bd344c3b84_5.npy',
     'fbe0cf4562ff0a8b4ed27e1169843fc2_11.npy',
     '2c481cabf7b35b91b54552910df2f7f5_5.npy',
     '58bd472927caee09fead65ba6e7b5d00_4.npy',
     '6322f8e665502c4aa9f2719d49300ede_17.npy',
     'f0c2cd1f58662d03799713d62174352f_12.npy',
     '12dd91fdb8d508937ac7972064cf44cc_21.npy',
     '503bb3ce3bcc6704a2d2291105d6f1b7_18.npy',
     'b914a76d4f3df7fcec152570159ce30b_2.npy',
     '1abb211df7c2e78797e88b88d4c263f3_5.npy',
     'b7eae8d5222c9445ea0c56b3d91cef19_10.npy',
     '6b76985d8d6da88416b133361c681f85_50.npy',
     '29ca45fa9bbc2f8e23c62c2506fff9ff_16.npy',
     '69cd1dfa915bdbb5f13f32e72a1e4ca9_12.npy',
     '5894178449035ad71d2a27909cbc88ba_32.npy',
     '112474a4a8731df01d7501e5d83770e5_13.npy',
     '7eb59111bda528069cf7466c06274fc1_34.npy',
     'd9892eebccba29ca659f3793283451cd_22.npy',
     '448a246711303e3ab33ad9e48a5e428d_16.npy',
     '71564292df648f2cc270617bbe3e61c3_97.npy',
     'ec120b5337b49cee0cc7f456b9341ec0_2.npy',
     'd9c8386c1bd16b024fc3db068306b2cc_37.npy',
     '9e04a48646c39f3587a2915405bb6283_35.npy',
     '32dc2fcb905ba303625aab8ee28015bb_11.npy',
     'dfbd9ea21e9240313e58756a3edefbac_27.npy',
     '1d2b186e5361be12ecac46ca4ca4ee81_13.npy',
     'e291e516414c22dbe0983cecc37543dd_3.npy',
     '20832883b94541bc1e07ac255e6ca509_8.npy',
     'a7ff2928de302e78c76de027f2f282e7_13.npy',
     '76f4b9ee3b93de263bf3f0e0b9c9e398_20.npy',
     '3629fb1fdf0b0845296e9bf9c37d97d4_2.npy',
     '3e6a079c4b9ab948499fcd175735da15_5.npy',
     '74032fb8e0ca017a401f28d8082b8b43_16.npy',
     'f7bd35eb24b5498cfb91b80bd440f102_33.npy',
     '55e759dbea171a310f80d55520f4e66d_3.npy',
     'a8356cd7b7bdb72d68d7d9efc2207ed1_9.npy',
     '9734548b798aa3497c3ad92e09d19b99_19.npy',
     'e3619c7778310cbb86ac1304befe678e_9.npy',
     '3b8cf6e6a3fd8b51978c6bfcb7985752_4.npy',
     '57693330923001bab06b6bbba360fa52_14.npy',
     '5abb93aa484232b9bdb78f00e13cc4e3_13.npy',
     'a285aabe72480dcd8506d39564f0938c_28.npy',
     '33c93b66d74238a3100edde5b3c941d8_23.npy',
     '12c4cca6d97bf9a9424cb22c07d44273_3.npy',
     '206461e1ace4da5140da1e1bbd377297_17.npy',
     '51f95e97b14a34a497b0fa8188e75386_33.npy',
     '6e72c7817634e2f4f5eafacedc7acc56_9.npy',
     '4de4a41772ea9377f1e8095f8c28de44_125.npy',
     '6828a18487aa0a6bc9493752c57a19db_10.npy',
     '8516a5698117ca09a775904d442ddd57_6.npy',
     'c8a2694584496160819e7df487f74b06_10.npy',
     'b85af07b26b163e373752a3c4b1591ba_2.npy',
     '643338b5f60106fd073d354eb7750181_1.npy',
     'db5b1901e4c998c97bad19a982042b99_15.npy',
     '275db385125088547e97f2cdd47d42f7_18.npy',
     'edfc03b45ff54707a62747cdc5de6031_1.npy',
     'de2bed23601eb1e788d260a660f7be0b_30.npy',
     'd52692beaed9df8475e10f25e3af0ee0_1.npy',
     'c186e81c4c6a6de304d18db8818589e3_13.npy',
     '0c1850983d30b22060c67bd671189699_8.npy',
     '4850a1d2cdff4b19db24989efaf48b10_20.npy',
     '182678722441224df758dbb9cb336afe_19.npy',
     '20f0ef8cd7e9378cd0f42b989d2368d7_70.npy',
     '94ccb14d0e0028cc597222b86505bb36_3.npy',
     '2429b71858735b0a56ca493cd1335d20_31.npy',
     'c99f5a6e6e92d8933ec2dc10f98568e6_28.npy',
     'a3e04d3b6156340afbca62186bd1d4a0_14.npy',
     '6eaa82b2f57d99fc9cb389f5921c1880_5.npy',
     '06d497198425511fc4814e011830356b_3.npy',
     'f9f280db0814612bac1c7ac76cf7c399_3.npy',
     'e1d2962aab28e2bf3ea1d922fb6a49b8_13.npy',
     '53694f8b491fda74b9166eb97cd9ff00_53.npy',
     'b9e757be2f7f7faff6c6fa2d8e26f682_32.npy',
     'c02c840f02b8d3e69791a48d0d559a80_58.npy',
     'f8f1238ee557f9f75cb3ebf1c16590a0_28.npy',
     '7264380a96f2b16576d3013acbdff899_34.npy',
     'd760f9779678ef35d0e88235f3be3465_0.npy',
     '0340d05d6398c750c06596bf2d995581_1.npy',
     'b22d5b7df0638d47f772f5443122c4c4_43.npy',
     '0d07d618dcde7638ca314cab7aa1425a_3.npy',
     '271d9a5977f012ca9b86453d0d0730e7_27.npy',
     '6ffdba35f00747b513cc8f86fc22d0d8_56.npy',
     '23ddd3ab745ff2fd4faf34f6c2327634_1.npy',
     '0f61ed1cd9ed4d693ee5d87c29b879b3_103.npy',
     '5625b54dfc58c527e38a16f5fe130a69_10.npy',
     '17e176fcdd04f86b59a64d855b801303_20.npy',
     'db484bfab1f6a7f18744c08edc581bee_7.npy',
     '0dd7f7fc702e58e3de5fc467d6c03fb3_8.npy',
     '1f505d3800cdd97d83234f9b70ebf3c8_2.npy',
     'f887218d3a78e12fd40f49d608455abb_13.npy',
     '67276781c7d53c39100c7a6b5208b589_42.npy',
     '700a9999415ba3130e2ad731515c3b1e_8.npy',
     '5ddfb038995eaed6991ed1a73d18d889_12.npy',
     '165a2372839b24c27e2028f023363580_10.npy',
     '767adbdc3b198cab5f91b73b972f3a8c_10.npy',
     '05f0805510d275ca5d95e06750de4568_8.npy',
     '245a1134bb924c1f68a30daf8fb5b40e_10.npy',
     'b0d513bed032cac002c42a3cae8ad445_11.npy',
     'fd39013459e7203bb5af036ab4da7188_10.npy',
     'fef7f38a1634ad88b89475b355ba69ee_6.npy',
     'e481944c6c7faee54e6a58436ab115c9_23.npy',
     '29aa0a1439296599d89c64720b2a31ca_4.npy',
     '8506187d09689efb0b3697dacebc8923_202.npy',
     '828702fe48e8f1d44fa470ac9daabb6e_263.npy',
     '67617806f78a97e3040b1cf88f749961_7.npy',
     '07b15883c12b5996054b54d103ac2f41_9.npy',
     '8c01b5daa14af698adb6419dd06b800f_31.npy',
     '93671abdd6bfb044178e0a9c68ec2a02_68.npy',
     '0e877d46cea960eadf676db4ef99961d_4.npy',
     'dfaf9bcb716bf3ceeb8a18585e0f270e_11.npy',
     '018f62bf91d6ab0df2b1b524e9219f01_45.npy',
     '6bda8f9c2f5184dcb36aa07206621578_4.npy',
     '30c3fabf6535968d7182fd92f99d8098_20.npy',
     '0042233408a9aac06d6cac4bc50c7539_15.npy',
     '1cc0198a45f5dd852321369efc46d9a2_1.npy',
     '1a6ec4e7e81bf9cbaf42ec27a340e331_37.npy',
     'be0a17c2c922d974a4abb955080cc19b_283.npy',
     '4ad444c87110069fcb4bb6d257341d3f_26.npy',
     'f0ad61becd1007bd237876a96c3ce33f_24.npy',
     '0e5f05b0c78024cfb604ce11f5c5f883_7.npy',
     '1b208b0985b9adafb31f8e6c97755101_75.npy',
     '27c873321dae9c20d0d57767b54e50d2_12.npy',
     '8ec2b4cca972365cac8884b25490c90b_27.npy',
     '83fde7747b0b0081932f0022ec9c78ff_38.npy',
     'fd30cd7bf7c2033492dfb2717c705209_10.npy',
     'baa9ffb502f72349606d89741a50d752_17.npy',
     '53ca4e7dc98cdc9cebecce59bfd062a9_72.npy',
     '448eebd4d412d2007f5fd409707a90f5_6.npy',
     'eb9d5937c8a93a6159a70e8908ab91e9_5.npy',
     '54d4e486bb20cad741a91d9cb64aa23c_59.npy',
     'e71479b4328de8719379e48f87f25d1b_15.npy',
     'b6ddec7c0e44048f566f67c2d4314eed_3.npy',
     '9a88ebf398fa73e8e7265f2c9b0dda20_41.npy',
     'e14c5d3271832e0c97dad571cda3a214_4.npy',
     'a33de3cfe1132fed4f35aa4a64802b61_91.npy',
     'bec54783f951db400595060895462171_11.npy',
     '81575fded506e94f9148a4dfc314c85b_29.npy',
     '6f59963b4d79ef88d8b5521582a01f14_23.npy',
     '486ebd719d1031053aa9dbb84daee77f_29.npy',
     'ad7f86c442c1cc5fd05af1203ad8db37_3.npy',
     '7894422a1c8a26599367017c4a629403_13.npy',
     'ee520cccf673c2f573a662f21216fde5_26.npy',
     'c064bc2419a9f0f1e24cd5de5e89e579_10.npy',
     'b763c7493726924f6fc2aa1621cd80f4_9.npy',
     'd0e086184b84ad0be74b13a517b16f4a_14.npy',
     '93974c3026fedfe9e6fe2a299f2f95d6_58.npy',
     'b552d3d8a54974194d17e1c7e209e678_3.npy',
     '19174e53394599bcdba4a0b599a000fa_23.npy',
     '4b863125bf6d6792f90c639eaa150bb1_2.npy',
     '22e9a192002e8412e08d4d40a826d00a_27.npy',
     '6fb6b103e54f52417e25c40989130504_45.npy',
     '0bdaecb03230b73c09ec83a0191f9509_6.npy',
     '07a2ba4711e2759e043dacf061fe983d_13.npy',
     '6ec14f028ed2e696734bdc511d9862ee_71.npy',
     'c574b6bd97df75d15b76b0c1ffb75a91_5.npy',
     '83df756d7fa4c9306bc9d56aace2c965_61.npy',
     'b15b042bb9b7cdb531b25ebde08a180e_7.npy',
     'cee8409255ecbee49c8b038da24eb441_22.npy',
     'fa618d1f3a6f9581dee5d656098e82ec_7.npy',
     '52c7047e40f82626d13ced99ffd7572d_4.npy',
     'e43b06c7b60133805ab1d74d155bc39a_12.npy',
     '3e6d57002256c871f9eb4f502a26c682_37.npy',
     '24ff985f4f29f940c2fd5f99e77f975d_12.npy',
     '81bfb5d793d5a5c6202435b2885df5ed_3.npy',
     '617f80b64526f6e0adbe6a86cbae07fe_24.npy',
     '9ed65763896c51ac44bbd1a2c75770f0_6.npy',
     '04bc56fa39bc915ffbb3acb4d4bd7fa5_23.npy',
     'ea3580d02e66212e081469d93f76ed3d_6.npy',
     '0fa0e7cda6e3c641b73b28b5ab09f96e_17.npy',
     'd1eb8e7b3c65a4be589b748162876523_16.npy',
     '9e5b29b2f08e8cbc8d101cdc43fa168c_2.npy',
     'b799b0549dac4819b3541c7c78394a30_12.npy',
     '9c736481e3f22f8243ea2e01ad3e9fda_6.npy',
     '72cbc39049890efec248cecb03524c86_15.npy',
     '751e759d4713d8527f838ae7e67c6b37_10.npy',
     'c9e303c56456e4dd32469988fded89f5_341.npy',
     '7ce47a15e5388ada4f619f97dde12f43_1.npy',
     'a7f7d0dc9f571169007bf46321b36a8d_18.npy',
     'fd575b3295a8b62d211e7e6994c7541c_37.npy',
     '7f8c4edacac8424730f96cd18d881ce2_13.npy',
     '2a8ad461ef8ad5f674de4328096315e7_72.npy',
     'ecde2a93a9cec1aad27bcc43669d1df4_22.npy',
     '6038b582debf8f46593e9665d80b0b43_96.npy',
     'f17fd14066e7ea024a8c59c480e87025_10.npy',
     'e491d41b8deab7925de12a18c184f90a_6.npy',
     'e0563fffcee5b3786c0ceb3013e72f1a_489.npy',
     '8bc6371e95a93c54b9f49207473d248e_3.npy',
     'cac66c9504cdc28076d8d43a38e727c1_15.npy',
     '73c627362e0710c249e16addeb75a6b7_11.npy',
     '3a98fdae914703396c1afc6c649dadb2_23.npy',
     'd2b2df65e55b62c85b8f25a1e6cb779f_6.npy',
     '1c47cb0844a50dfd81444fe95cec1d05_9.npy',
     'b795364f56314c205692c9996f2b57f6_12.npy',
     '04f47da81d6ad8d609ea9d93933ba96d_14.npy',
     '348592a46e139d6c6511833e73a2cf6c_8.npy',
     'b36b91132e5591310a2ee55a1cf8d43c_3.npy',
     '87109cc6bcbadede171e58967910fffd_6.npy',
     '239479682be67905bc16cdbc2ccaa4f6_17.npy',
     '4946ebf29ee30e94614b5e616707ab8c_1.npy',
     'e5a74e6fe32747fa9c1d7eacc698684c_26.npy',
     'c128a31d29ecd64055f020ba533be887_31.npy',
     'd8d1d42443c057141ed4def85bfa816f_1.npy',
     'd538f44e89366539b160303b9d564d73_26.npy',
     'ec6bd756b683ff57122ddbfe38023371_39.npy',
     '00c690bb95f95f6a33ec2e5799bc3782_1.npy',
     'e800cf062a47405c71145c8a6ef60b34_11.npy',
     '9533faf73fec89e366e6537dbb161cbe_17.npy',
     '8f2773e4bd11034729a96055be7c0020_3.npy',
     '5e0e42dcfef31c4c5a85d41ef2feba11_1.npy',
     '81e4e50c94e8c61b6759ee8609f364e1_48.npy',
     '173f36228a749356ccec11cbd3f64c23_14.npy',
     '36fc579fbb7d78c50544e4107ddd617d_12.npy',
     '5458a1248ab80595ebb3bce9f439748c_19.npy',
     'a470836f0d818859bd64b1c673a3f281_12.npy',
     'e7f7b9b4754795d4abe234b8d9cf2606_21.npy',
     '7f8faf8f3ccc3aaa4fe994bdc7fdbb88_4.npy',
     '28bf6cf681e2917671a7ce9033c07bc4_3.npy',
     '59f8c6092b1ea04405914f3a705a9115_341.npy',
     '1897070b2af0bab1f45b5966f77961d0_3.npy',
     '80442a936969b6e5a100f7ac7b7ff47c_3.npy',
     '66320afb9f98779e08d1dee7417f4dab_12.npy',
     '593a9afe399281ecea617edaaf1575c9_12.npy',
     '34b551089fa151272cdd0176df4bdd1d_7.npy',
     '47835c5da18a4976cb5002783ba91fab_30.npy',
     '3265583e0634f40830c44ba141768c2a_29.npy',
     '5d9d378dccdc0ed2d57fc7969739d03e_8.npy',
     '383a50274645c80b212fbc511ba4ed1d_2.npy',
     'c853551a28c45a645fbf4ce9162bf450_4.npy',
     '19f2d7dca4f85fdb4150abc2bc9a5cd1_2.npy',
     'ddefc8a5542d4fc41fd7cd1e3ae37048_2.npy',
     '0f0c989bdb118d8f4d57a2f43fec0bd3_2.npy',
     'b9e674b417bdb6270e877f957a15a611_25.npy',
     'f8081159ecd1683e8c89d4f7e66b2859_2.npy',
     'b4c2297b211d2a25a7339d067cf6bd17_10.npy',
     '838b221e6564ad88877546178b423238_18.npy',
     '6b5c33e4d103f744dd4eae577d1d28ee_11.npy',
     'ae8ca530a72690bcedd2d026ce52e498_8.npy',
     '8a52049737b19b11b377e5ec1f8d01df_47.npy',
     'a22f064430572e06ce37b6d75dd32ed8_16.npy',
     '47cb512a1e48c23bbd901464ecad0932_3.npy',
     '6e44258944447c8e0ffa716c2d44d50e_23.npy',
     'a31549802dcc6304ee24f589cf5480a7_6.npy',
     'a50ff4675de6055bf287561fd3e0acba_1.npy',
     '4d9b09e9e4a0bb9c2956fce1b243ba7a_116.npy',
     'dfc46e276adf7c208a4017f96b1cba82_21.npy',
     'e768bc5c78155e4b8b68709c4e80b09a_8.npy',
     'c41b86d1c9dae593980dfa7ca358ef3e_33.npy',
     '0856447a3c70217feb88721e565496a4_8.npy',
     '8e8f8efba2c20c69a0231733300dcfbf_8.npy',
     'b59a1bed7af72fc8a2643d21bcbccb5e_4.npy',
     'afa63fc08d86c4f4a494c163320b9ab4_2.npy',
     'a4ed0616a0a33f0e84c7a7f84ffeb9c2_37.npy',
     'ba4e409052a1b046b560fef416b6f492_76.npy',
     '27d1062501f64b9f6854eec6f8faf8c4_8.npy',
     'd4ef2d6d19d585fed601c561c1363e40_12.npy',
     'aa42d6f3158f50f5a4f247e4722c1a19_23.npy',
     '38e3264eff55a577fb745231b952b0ee_4.npy',
     'd5fce8a14510a9807bcb98ae36b317af_12.npy',
     '9ae40a9819a2426af700df9ff24fd414_2.npy',
     'e68dae52adff6a6efd9acaa14dc409f9_21.npy',
     '79cb3808e4336b48be9f195a1b10745d_132.npy',
     'c5d5b387d7fd2d9cc949587943487580_27.npy',
     '09088e0dacd0c6c0f41ed693b1580039_17.npy',
     '76c1ac1dfb40ed97932d19c9e5a6f881_137.npy',
     '0493ff1e15f3093fd367c199c0cce8e5_6.npy',
     '0dbbf142e99d92ba4446e6bb19703ea9_4.npy',
     'eeb182695b3cdac2ca21ce9c9a76c5f7_10.npy',
     'e22a59c740bad26e22b51eab1bcd6611_10.npy',
     '5658b2249692021104620f6b9f5847d6_1.npy',
     'cd77b09a10b0c825b028d25543887a63_10.npy',
     '49203b56bc067885d62603a5adada4ef_5.npy',
     '4e4c6cece853101b9ef9f3e3796a5293_3.npy',
     'f1346d22a90b0b3b1ab380be294aafb7_27.npy',
     '20d8202a6a4c3234c963b11501d1b451_85.npy',
     '02ea9124df6f17393be5e4b208f0560c_16.npy',
     'a7629197613a912901ddf009823b46cf_29.npy',
     'c24a7c43b452e4ae2cf7d4eb36e3e6fa_18.npy',
     '7e2bcb34c0aa2f8d938da0d055297f1e_8.npy',
     '23274f4cec18e7f67cfd1110c911bd20_26.npy',
     'e656b771d6ee98e148fa3497c2a25b96_10.npy',
     'ee6bfd04d32328ea4def7fc145323623_30.npy',
     'ed98aee769856a50c7a9bc73123d3031_1.npy',
     '9f4ae3323447f44ed3daa1fc8de39721_33.npy',
     'f65a718155b49f42dd70f8df7eab504b_18.npy',
     '6db8e20bd82b5a1d4316dd36a679867a_17.npy',
     '5355aae3d3cb596f93b2b1c36e00d32b_3.npy',
     '5ea1f55cc94a934193f5c36f0a86744a_11.npy',
     '11f5f4b847aebb43f0ab326f818edeaf_50.npy',
     'acad15209187f4159c81e2122562508d_219.npy',
     '2f6fcf8332b67a641210d2c3d27fad6f_129.npy',
     'd35eb7a671b0503d7f901a6e7baf63e5_24.npy',
     '42cdc119ddd081fd2d4af6c2e6c07519_4.npy',
     'a6009509a0f656311a341611fbd4f882_18.npy',
     '8a7e18a77edcb07bca176c4b48a8b49f_38.npy',
     'b2d758533156e047c5d87e3360cd26dc_56.npy',
     '447e3cb7886c2dc564ba9ee818254435_27.npy',
     '059736441ba1115cd7d06d04d0765c9d_11.npy',
     '297678d6c8d488dcd5ceeb5d343e52bf_3.npy',
     'a1d1b1d0f03cceed47a51a2481a3ab5e_4.npy',
     'a0d6d9b9dd44e93b651ac077e177ccef_63.npy',
     '006a1f0cfc00e4b7e3d74371aefc533e_118.npy',
     'f79baf50fd66a7b882fbf02fae2653d5_13.npy',
     '1f326e04ea5a592a56b222a29ce96919_8.npy',
     '4766620ece1378ab367f41d9e2975c41_115.npy',
     'ed95500276c88a1172635f445501ce93_2.npy',
     '38e8376d4a3e82c59d04d7698131f4ed_6.npy',
     '2abfefef18cdc989c535200bf2e3b2aa_1.npy',
     '70addaf19f95739a3d79cfbd42f34e15_7.npy',
     '0b7c9f1fc9c0756c172f2d20faa889ef_3.npy',
     '5096b2a4623a09009ca125c2c2bff69e_18.npy',
     '72fda42686da8101e9a5497efb84f043_11.npy',
     '09cf82877988331ddc6a52e106bd474d_110.npy',
     'd2b51e47f47c2663e647228dc7eaf76c_34.npy',
     'b502a1cafe0224ce7ef1f7e617745c54_2.npy',
     '0eb79e13104412c7f7085e7dabcae847_3.npy',
     '943bcb94ab57a76634f20c311f75811e_9.npy',
     'a8349dfa61760bdaad538c35edcf8704_20.npy',
     '30af40ebee901b39345068ca161f4f80_9.npy',
     '8b33af060e0d24cb1857db8a4e17a41b_3.npy',
     '2f0ebf7b6faf4405596d3dd00e45999b_2.npy',
     '0aa77d5b82678117df1821a912dd1bec_13.npy',
     '6939619f4c777b75c67ac2908ca1e2a3_19.npy',
     '57a64bd25138f40e2212c0695f60db3f_139.npy',
     'c9501596e416a22b10bbea892a7aab52_11.npy',
     '1e3d732f32506ec6c8595d2b2a5fdb1e_4.npy',
     'e3cd95ebafe6618bb2e4f93a552ee7fb_12.npy',
     'a7c4451d5cf6c3af4a9831e40b389fe3_40.npy',
     'c918cf4c6a16f320d2ad781dfc95b7cf_3.npy',
     '97c2ede4dbf96ffe16753fffd1b12a65_0.npy',
     '94c01dc20db815a290f2abb28ae44c01_82.npy',
     '927a4f793cc3e624d4ad22caabb790bd_11.npy',
     'b61787aaaaeeb7db611993d865211e92_12.npy',
     'd1b1992a8a80dac28ca9182ca5da709a_10.npy',
     '4dadaff74800b38e9ae6d15a6e9db3b3_3.npy',
     'fad5f40dccc49f85e4007346b6e2709f_4.npy',
     '92048cc752bfda54835bf9d077b2c9e5_7.npy',
     'aab1633349ac128d59161b3b65a39173_3.npy',
     'fd4d14803b3bd6686ce0167a87fdca58_1.npy',
     '5fcef40ece2422e3ed12ef16cab21956_7.npy',
     'ff44acead635847e5baa40bf193810a5_24.npy',
     'fac2e1bf8e4cbb01f7462628dedd8c1d_45.npy',
     '71cd65f8c8bd7e7ac466ebfdfc73b193_7.npy',
     'f72f621bbccb9a39ada376e3a11fe772_2.npy',
     '4610c975dbbce7ef2280cbb4775f4fd1_28.npy',
     '76e817532921718ee65025052fc6244b_11.npy',
     '6b9ee1c574e72e3b3c5485b95ddfefb6_2.npy',
     '96604ffa343c0d481ff1906c5f097136_19.npy',
     '1eecab6dd4cf2c75c8143f54245aea2d_17.npy',
     'aff03440397ee5ecbf846eba0763591b_2.npy',
     '25b72c29d284c31bc2e4cdd9f41cf375_41.npy',
     'b51a910398a26b563ef37aeaf584314d_44.npy',
     '4cda4890b80e18ebb9543f6ac60e5102_4.npy',
     '8cadf1d51c386c60139a3ce55f3310cf_5.npy',
     'd3c13f33698751913c4d24e7e51a84c0_5.npy',
     '562845c9f011c65a5be7413763e3d284_3.npy',
     '00df2968fe0a31809ac1960355871726_32.npy',
     '4e5835071b8989213ba6db3b9a867dd2_9.npy',
     'aee1c172ae671873e39fbe92c73c2499_91.npy',
     '109d20834eb2cbaa9623ff9210dc75e7_22.npy',
     '0ff9fe1d16f39f9b3b103221907ff24d_25.npy',
     '722b87d91a492665da85f9b7cbcf5c80_1.npy',
     '4d3e4f87e1c305782a116fe2d9c7ff63_2.npy',
     '66765e4a1931cda6de63d71900b64fd4_24.npy',
     '810e0aa0f5f7229af3fb0f0f2a9002f5_2.npy',
     'b75365ba10a1109b61d2db0d9dd03647_24.npy',
     '885ed30b317c3fc3a2f3a6ec932c029b_5.npy',
     '1cb1fab5bd98e79fbb14bdc966226aaf_14.npy',
     '761869203339a4627f69f61a69e32d57_15.npy',
     'b706c0c364c38e250ca2875c574e4c94_17.npy',
     'cc13f9069babd38e600ebf5b4cb487f3_9.npy',
     '703fbcdfab1e457284543bf869947909_11.npy',
     '3d049b6d1e1595a0564928f46c451b8e_54.npy',
     '1f326e04ea5a592a56b222a29ce96919_14.npy',
     'a9445c4a6e7b1ff35a277453618bfdf6_6.npy',
     'ccc1d24f3d30945ab8d76cdafea8a9e4_8.npy',
     'e1f8d7555b21a9792441bca661ddf476_11.npy',
     'b827222223a0f37d422277f10b7c9966_20.npy',
     'f36c67652bbfee9a50d575efd11dc21b_11.npy',
     '1fced69a6893c9ecd07a105cbf4a9588_2.npy',
     'c3cebf0f885d7706cf36a23c5fee143a_14.npy',
     'eecacb30cdf8024419e8115d806940c7_47.npy',
     '6d3c194342cf74c509a04d0e0648829a_270.npy',
     '387ff025fa29be0b0c7a6faf9ea2f65f_12.npy',
     '1ac11f01b72f2e41d40e6b4515952152_5.npy',
     '30cd2cf73a636b804a930df17c623047_9.npy',
     '49999f8b1b11dab006fb79822b90e515_13.npy',
     '46942638732f3b91f1bbe83bf5d62d3e_7.npy',
     'eeef1cf7b310e4640fbd3a049c0c3bd4_17.npy',
     '9c6b5d4bdc58ddcbdc43ae9c90ee9996_29.npy',
     'db477e14764a6793454db6c4913b59c0_3.npy',
     '225a4d52fda1d5597a96cb7cc1f1eaab_78.npy',
     'cd1a3a912f2767fb862907ec5a5239ac_76.npy',
     'b63b382bac709d023137add9385534b7_6.npy',
     '836f14a1f38d9c12217d34c9012c805c_73.npy',
     '1bf30722776c8c9b4c389f7b60503122_17.npy',
     '1cdaf0c8f7c6cc534312d5c1aedaf353_13.npy',
     '12ccd5a823f0d7bd4153aee4b2f29f4a_5.npy',
     '8cf6fc429a158f156c1ab3a27030f561_21.npy',
     'b9d40f24d7e8a3e70d0d1e6bf24fa7e3_4.npy',
     '837e57fdd3ea2edc9a4cbec1713f296a_21.npy',
     '97089324137a2fe63a39fc665565da82_37.npy',
     '3f0824808775da2b2913d2f426b29b1d_1.npy',
     '06f97236579f3b39dd353b076cec6dd9_5.npy',
     '26a8ddf923cee5e81c42944cb7e6288a_3.npy',
     '363e733c360327996384d8b1bc4c5222_20.npy',
     'da18080f4545128f314c7a580a39c151_21.npy',
     '6d804fc66a4692a0c05167ce24546914_22.npy',
     '48c40a6f1c851feac93329834800f61f_8.npy',
     '206d531e618e77d19eaccf53a067a31e_2.npy',
     '03b5fda598165779c56654a649de7a4f_14.npy',
     'f5b6a24fb52646ff65cedf8836b8d079_2.npy',
     '6b9bdc5dbe5e582c8e2df1b04435914f_22.npy',
     '35637877e64b1f8722e0fb5727294eaf_188.npy',
     'e62d216610dc7083b1fd8f953c28c375_24.npy',
     '162b863a43ba9488f6fba339f57e1a46_1.npy',
     'fec93b60afe33b623bca8b38fbfa1ed6_1.npy',
     '07160924d52d2032a6b59c5df983ae29_6.npy',
     'b72499c6ccd1696bb10cfcc3cd98818b_13.npy',
     '24a0a30cfdf986bca68852e72c6edb1b_63.npy',
     '2a92936cba7ad145efc9e9853ab99c2a_7.npy',
     'd7965a443207858a72558ca8ae671b81_14.npy',
     '32323ba02c1dcb6a11af1de752889209_11.npy',
     '42236960294867ca6359d6fcb8ee45a9_11.npy',
     '696c764c670d4417db1bc9968cda231e_5.npy',
     'f594c9cbb3202100537e4ae372e17c9c_7.npy',
     '9079692aca81c0dd6195639e0e31e71d_14.npy',
     '2c5e4432c1b2c2c340fcaec8dd6a073b_28.npy',
     'baae757adf99291c515fb28606c597c1_2.npy',
     '90444e510bb2216982970dacd4be644a_5.npy',
     '912580039ae2e711a2a7afc8f1ba4591_39.npy',
     '9256b9439ef8ee769d876021f34d2750_18.npy',
     '49d1fedc2845d96f59c6bdcaa6da09e9_23.npy',
     '5625e64b856bbf6b1945606b10831d3e_62.npy',
     '94fc0091d5bebb570bca6d454b886763_13.npy',
     '651c740dadd592ba77e70ee91dcd6fc4_5.npy',
     '22f330763cdec47cbe65ec4927e8d543_16.npy',
     '96604ffa343c0d481ff1906c5f097136_4.npy',
     '5067ad7f162d8f71e39cff0838936d9b_1.npy',
     'f63d999b4983e6b011158e4bfe54a951_3.npy',
     '247a43fa2f2df9999cd44a965cacd9b0_4.npy',
     '3cdf17fe1b567ab8625bb43556348486_6.npy',
     'eaf9af9804defe566cef354fe37e0da3_26.npy',
     'ace5ce8210ae824d4f547a7aa77484c9_7.npy',
     '9a3f3f7294ec69e1f6f0e40d037f61a3_22.npy',
     '5afe9613d23c12de97dd18ec9c7908ab_12.npy',
     '9ce597aa6d0b493ac316b54b17e837f7_53.npy',
     '289a1406b0bc67a39bd012c9060687e3_38.npy',
     '260036bf15e5f883bad9b5baf1658672_11.npy',
     '375a4b2a510f7fe75bb8382bdad09f5b_8.npy',
     'b042666daa8314fdc050f636eb4b0a6b_15.npy',
     '759b00e3cc7c394531edb36f37af8ba7_5.npy',
     'f999db86954a06220856226e5f36ef96_3.npy',
     'e988fbf8d5812c492f39c53173789448_50.npy',
     '86d2cc5ce7de608793973b4c4e26da80_10.npy',
     '51243c6f1a424f74440ee5e27e367e9f_11.npy',
     'ea0d453f3cb670e4bbb31f2d5ab6a242_1.npy',
     '8a8468f32bec3ea3ee20ccce81e0a48f_14.npy',
     'ceff0e277c2ea257396e79b6e0ccc838_11.npy',
     'd50958d38149e234d201b988f4f313dc_20.npy',
     '5b335401391927dbb452ff08a2535097_14.npy',
     '655cc9717818bdcb719ee3bccbe83751_17.npy',
     '7294b875a1fb188816a67869d40c226a_15.npy',
     '9ccfa71d521532d2eeb30b2dfa015e3b_6.npy',
     'c8dc2c35dc9ecc141c5f2a357e32a28d_14.npy',
     '81c6b7c3131b58fb418420d1da6369b6_70.npy',
     'cd193f8b6d5183146cd7629be095df3b_11.npy',
     '4632e408d722169af7c51a4b3054857a_75.npy',
     '7b4c34064c1093de0157e3fa4789c5c6_17.npy',
     '2d9011f1342a2c45aebca0ba56a093b1_2.npy',
     '5492433d14436c220dba636cc2b42a4b_7.npy',
     'c4e5d8efab05333e2c3711eaf1e1c088_4.npy',
     '149870b03aa0a003a5ab3e5f2f125d9b_142.npy',
     'dd2b63e5d6e8c6ccf9da737ec13e54f6_32.npy',
     '4bc1a06c666cb2b2a1e9d7bd3d177392_2.npy',
     'bc2bec5602937e48589c100c7787a037_29.npy',
     'c5503550bfd099cd0a433a88b0c5c425_4.npy',
     'c1c3e9aa25f798971a250e62425ba5c1_18.npy',
     'a237401b2818bc115b1ad3f701522592_23.npy',
     'ca12b42016cdaab01dd392daebfac276_43.npy',
     '0fcee8b3d8eaa8653e6c116641976f78_15.npy',
     '7bff7e378a10771faa91ce7cd60a036d_3.npy',
     'eeea1c0c5102c005a0796fda4cfe8110_5.npy',
     '1d02e1de263f82c4fb34d91bc4dca1da_19.npy',
     '09196de4d25982e7036ad60745f481e6_7.npy',
     '3f205f09ba8f4a496ae880105a7c11de_23.npy',
     '986c0095530b610d307c72efc31e5c6f_33.npy',
     '2f8e547979112611a24b566aee3fa274_31.npy',
     '12fcdd5a2e25cbd3be4ec665b40ddb7f_3.npy',
     '16500c61ce1639d33e09c071b09e9619_4.npy',
     '93fd3940d8fe7c596173897dc53eb0fc_19.npy',
     '235e9433958d8cb791bcb3bf0a8e59eb_9.npy',
     '27f0b4a0efa7c2d182362e4806d1e8b9_26.npy',
     '1d24f742c46b44773db90bea9e3d2cac_27.npy',
     '271d9a5977f012ca9b86453d0d0730e7_84.npy',
     '4dd00005c6a75a7fb775d577da6a8536_9.npy',
     'b48e10a74c115ae76cd8d88685a8920a_8.npy',
     '952e433f7bc217555256911d19f051cb_24.npy',
     '8bdbb62d4a5163ea588e682be7227cc8_9.npy',
     '2009daccab0a68e75509992673f7fbc9_2.npy',
     '98c2202fbf1bc4db9730f7e501200795_3.npy',
     '8a0f503fb267d4537c37f1119c3451ef_12.npy',
     '83db706d90d9175761d60edad0b2da37_23.npy',
     'a3794caa2b6a1c62b1f1c6164bc92d62_28.npy',
     '7884d2c1c121bebbdac5335b7a81ee05_3.npy',
     'cca75c306404dc541d90da92a2d9d327_3.npy',
     '566e13dad60e68d8648fb4fe2e51c4ee_8.npy',
     'f277b238277f3fc12965e476a0125774_17.npy',
     '27f89f54e246b8dcdcf748cbf0309f75_2.npy',
     'e18aa7d342d0342df2d2c1b99b233f20_4.npy',
     '348d1d24f2aa7a466191ff6ecfaac9bb_6.npy',
     '648986b416834b5cce2291e741623715_25.npy',
     '273eeb35b368b1a4ceca8d45f509f2eb_30.npy',
     '9cae574a6f00cde021a4c86495841ca3_6.npy',
     'c8fced5366c85b344fd61b1acd8f9202_79.npy',
     '2d28de0b99c3885f21533168a88b6d57_107.npy',
     '03c2a89866067e8fcc3c4facf9888d72_21.npy',
     '8f2de5d769a57aa2392c88abe744f7fe_1.npy',
     '7676017b69813fecc0665e8f1b0733b0_67.npy',
     'f504f24a6ea8866748a573b9584733dc_3.npy',
     '822a5515a3a02d7bb444c4cd3d3eed7f_3.npy',
     'b5b32fbb3e04da4449b6b0ff5f6e1383_114.npy',
     'eaf7d3f6a9936536b50c8c1d596ff009_6.npy',
     'ae3c127fbff3f64ec3f8739bc5e740a4_9.npy',
     'd186c3e3779d9e2ed9ec64dbb50b4563_0.npy',
     '80fefd21d178c20c04f19557b8c53a09_2.npy',
     'cfb50546523067a60fa3a4ebdacca0b9_2.npy',
     'cba7ae80c59bc2c718002a27b7a3ef35_144.npy',
     'abc26f072d49f18d00b0023f79f092ad_13.npy',
     '14ad464333de4924f6b01c8c4b0f9251_18.npy',
     '936ca33db9ab9e71f41a47841f13beb5_7.npy',
     '401f87e27f3ef5c3862995c5426e09f4_40.npy',
     '013ce82da613a8fe5b438e7ebb09e9bc_4.npy',
     '1a03261d2ca2fb1d5dc1a1660557aded_35.npy',
     'e85bf45839080363994c81dd238bda29_33.npy',
     '73c3346fd575ba399d33b10b3c82399b_13.npy',
     'aaa8200d5f3fcc34b509ee4ae14f51e5_11.npy',
     '7582d62b158f7adb206acdba5fefe597_1.npy',
     '64e0324ea9f438ec31184e0f86ea85ee_5.npy',
     'e4bbee7ec153d18fcca8d8a474606cc7_27.npy',
     '9a2945ca5e75af1f2255e4640fc75847_2.npy',
     'f8eb6fb8188ee4edccd5ca899ff51833_3.npy',
     '4d3486a0b82cf0c50a09685a90083258_6.npy',
     '8c6671871dc0152a138bf055b72c0634_179.npy',
     '751dc8db6472ca89248072b0381311d8_47.npy',
     '8f404ce99cbc95cdc3ec35d51ade7dcf_28.npy',
     'b9e904d62918935420bbc328bcfa0630_17.npy',
     'ccbd2ca63743ef227c3ecbcb2e82047e_40.npy',
     'f4a6fa62e71b641176b05a7f9cbdfcfa_14.npy',
     '1dfdc4d17e387f1a3e8514a1d3e98b12_213.npy',
     '1d920d6f92fab97cb966018f9b7f11d2_19.npy',
     'fe940ea6dad6765bf65358546dec1fa2_22.npy',
     'f444fda85597bcf56f7d6ce12d2584f4_7.npy',
     'f6682927953039d399e35ce84ea45e7a_29.npy',
     '519af903f91d06fb35c4676bfc770450_169.npy',
     '635d0b56477e7ee9ee342aa018b079d6_18.npy',
     '0658b7230522e964e7445a617c99c16a_1.npy',
     '386f89d8162de93e765d5b73a84e8abb_21.npy',
     '5071895a2fb9d0823ba6c64ebb5be6db_7.npy',
     'd081b8e05bc371b755dbfaa3547545e3_35.npy',
     'bb88b2e26c4b8b982cb9f2212247c307_1.npy',
     '509dfb9157d13b55edf8dcc93ca9e284_4.npy',
     'ab4a9f6311970e5c14c555c3ae7b4f2c_16.npy',
     'd9ccf910cf09b0d23951df41ed46c918_3.npy',
     'f93ecd3a65cc6aebda89cee46b31076b_18.npy',
     '5e84e5a71afe7dfc81650aab51a44e3b_12.npy',
     'bfda9b864a076c3f92501d78eee3590c_1.npy',
     '3475a95e9ffb4fb96b6aa85381feddf8_11.npy',
     '9dae9dd33a7fcd4e15a8d8755809c385_6.npy',
     '1f57462b9cd7e182acf5a67606caf976_122.npy',
     'da45fe1c509edeac52c0368f69df9091_60.npy',
     '34163961c58d138f2359ae902ed2b20e_75.npy',
     'b26c52e919907734ba2cc4e5ff9ac094_4.npy',
     'e1e271c5dd11bf49f1db53a1e34f2ac7_190.npy',
     '3e085f8ec8de894e7299c65115043d51_11.npy',
     'a12364e63da6d836a61af50b633c675e_5.npy',
     'c50f641ba7dbc7e95f95ce10fdb053b8_5.npy',
     '1dbd5ecfbc1a4959c9862065b3f6a08f_6.npy',
     '3ac52c445a63ee89b07a46485651eaef_24.npy',
     '422249da64a85648704e6e843c963d2d_16.npy',
     '79c14fca9baff3a956011ca2a75dfdcd_10.npy',
     '48aeb8d40f020de41f0805d333319f0c_9.npy',
     '9604c138483482ca61b43e04de7899a9_33.npy',
     '85e9e9015fdffc2a6b44160b332a9e37_53.npy',
     '81392a26f9df792df54d9e9381593fd7_5.npy',
     '73b44caab43002ccc83dfd26f04be005_13.npy',
     '3c333d7bd1e99b8e8ba6e7f8d907679c_18.npy',
     '6989eadf3a2cd6a9a09372e9db3f74f2_8.npy',
     '48d3d129ff932c52e7b8bebe1b5c5126_7.npy',
     '76f7a4342ed80a39c78506a5951706de_37.npy',
     'a69356a07c029c2155c927ae3c1a6d11_49.npy',
     '5653c8fce8057bcfe440841a8b5ca2c7_1.npy',
     'b2642541479d4476c71781b03cbe0d87_26.npy',
     '8e0298070c84e6a17fc00303fd716ecf_14.npy',
     'd2dfb319b8339ad0ad7c7f12561f77ee_27.npy',
     'a8334184d48b05d4fb73a9fc181ca281_2.npy',
     'c28c9970318669ae8a183d2f0a314de5_16.npy',
     '01e5409c04aaf2bb7cf04a8559540c45_32.npy',
     'a298cac5f4935139c286634d6eac617d_23.npy',
     '911b5a00e080cc47351232b50e9b34d6_34.npy',
     '51320e3b2160e015ffa6d6302da2f960_47.npy',
     'ec120b5337b49cee0cc7f456b9341ec0_20.npy',
     'f725c8f61367042c9e9e458c314d63b4_11.npy',
     'fd5e2f6ae1e1079524ae3a9ed12074f6_3.npy',
     '2884cafa3cab053e4c047bc3dfc3fad1_33.npy',
     '0aeb213e17afdb8b2edb1c0596ff6320_6.npy',
     '759f1bffeec7a1d8bcf879b01419190d_80.npy',
     '01f4a74dc711666a483f7ff02454dedd_13.npy',
     '603547e334e9aa1197809ddfb4592e52_149.npy',
     'e7f6507e703ae7f6c143f4d842611d4b_14.npy',
     '8b1572f8e786f8e6100ff9955eea084e_40.npy',
     '5c241cdd0eac19a83aad881b9aece29b_26.npy',
     'b4f35f206b844b3cf1cc77203f36a04c_41.npy',
     '2e25038cda77712d10a0469388db5150_4.npy',
     'bd5e3cd9aa9689692bd0a991970ab922_16.npy',
     '778a08cdcd6718b73373d7d5fa7c3104_9.npy',
     'b0648008b453bf3a7d59aee86242c06a_65.npy',
     'e8168a2d2212caa16285ffc565817571_5.npy',
     'cb61419a92fb211b206587bed71f9469_23.npy',
     '3ed26c84b45a7354896ae3469f5c5f20_4.npy',
     '34c92a8b5e7be3f6a8cea4921311eed8_29.npy',
     '92ae6f44e17c29835a100544a9948e0b_77.npy',
     'f0cc1b5099d1e1b54afa62243d4f6ba2_66.npy',
     '1c5d75a08780652ed36d46d4037ab650_11.npy',
     'ab936c84515454e635c505606186e562_21.npy',
     '4fca2ae4c3e84bf8c8eb03e21c2a2019_5.npy',
     '7d93dbd6565e168da02ad91183111319_66.npy',
     '3b4227b3c7b745fb42188c80c5f4d0e1_7.npy',
     '9cbf649b7aab06c9cd6c1ac82eda2706_16.npy',
     '11ee2872033fb8f9f64aab425190f380_4.npy',
     '833caae3ece736f7630d92fcbe69b814_27.npy',
     '728e5fecfe48b67395544334f61eb36c_11.npy',
     '3d09dce3c9851eed16bf60b509607efd_27.npy',
     '88e4a379bf5ee7307f129349ec0d1744_69.npy',
     'a4b7678e407873d98000ddfb87d2315d_13.npy',
     'bf47a02e1080e6ab24a852bd509b2844_27.npy',
     '91cd3b4b77614cece23d135358238542_19.npy',
     'bf3b51c47123491dccc58755a5b515b1_53.npy',
     'f28b91f412703d9923bd6f342d7f2aef_15.npy',
     '14ca01a113ec78679702d6f14d6e9abb_17.npy',
     'c9fd70e85708364c8520901fc468f872_18.npy',
     '6e812d1daf6e3766dabf6fa7daf56e59_36.npy',
     '40ff9522009a27553cd1e1488973a4e3_6.npy',
     '3616925e08c68460de819aab7c870e26_7.npy',
     '92e1810f16374d3aa3e27ca991a0f789_6.npy',
     '864a53ef5606fd201b242c371e7ea545_15.npy',
     '94c277922be20273f695d9245f4f3ed2_12.npy',
     '16301c006504d2e2bfb684c794d94171_26.npy',
     '0233ec9c394bdf65690c100b108036ed_14.npy',
     '8473d1f666fe5e50cc1fe8bbb02438bf_0.npy',
     'ddb87e364ff4b7d7797c4c89c056eb12_34.npy',
     '50e1a23014a6b8b100226dc58d6fc816_8.npy',
     'ab87ceb676ce194097b3ec6101449f7b_13.npy',
     '2ce044ab805ce531d656840782a7994d_18.npy',
     '60e215e9c769e1d210b2dfe343665ac4_16.npy',
     '91619dbfcb573435deb916d453128a10_28.npy',
     '44f3c8c90a7fc6aa47c0a4febfde83b3_37.npy',
     '7c36fc467e26d99adc2a0bba6935617b_3.npy',
     'bf0a70810a69ca3630e44e9b7a617d31_31.npy',
     '9540ed1f2467c78a27d32cb7954b33b5_3.npy',
     'd337133e10b7cbc5e60a1b700520c9df_18.npy',
     '22f03f9af0b472f9ff9fb7a2a9f0b977_107.npy',
     '5ee82ed61f1e1275efd2117addf1b82a_29.npy',
     '138219448ec5bf37d513ee7f940eb1cd_7.npy',
     'fc2ac52589f8e922f61303f41b31082b_22.npy',
     '7e39387312723c0515b56bbdfe340b27_66.npy',
     '5f2d448d605f06b2ced6e279b30d70f4_26.npy',
     'a0798ad2a218c9feba2d4410cb681c60_21.npy',
     ...]




```python
d3 = donut2
for i in range(3):
  d3 = np.diff(d3, n=1, axis=i%2)
plt.imshow(d3)
print(d3)
```

    [[ 0  1  0 ...  0  0 -1]
     [ 0  0  1 ...  1 -1  0]
     [ 0  0 -1 ... -1  1  0]
     ...
     [ 0  0  0 ... -1  0  0]
     [ 0  0 -1 ... -1  1  0]
     [ 0  0  1 ...  1 -1  0]]



    
![png](video-compression-old_files/video-compression-old_6_1.png)
    



```python
xx
circle
```




    array([[20000, 19801, 19604, ..., 19409, 19604, 19801],
           [19801, 19602, 19405, ..., 19210, 19405, 19602],
           [19604, 19405, 19208, ..., 19013, 19208, 19405],
           ...,
           [19409, 19210, 19013, ..., 18818, 19013, 19210],
           [19604, 19405, 19208, ..., 19013, 19208, 19405],
           [19801, 19602, 19405, ..., 19210, 19405, 19602]])




```python
CompressedData = Tuple[np.ndarray, np.ndarray, np.ndarray]
```


```python
np.concatenate([[[1, 2, 3], [7, 8, 9]], [[4, 5, 6]]], axis=0)
```




    array([[1, 2, 3],
           [7, 8, 9],
           [4, 5, 6]])




```python
np.array([1, 2, 3]).argmax()
```




    2




```python
# @numba.njit
def compression_round(data: np.ndarray, d: np.ndarray, dims: int, sample_ratio: float = 1.0) -> CompressedData:
  # counts = Counter(itertools.chain.from_iterable(for (a, b) in itertools.product()))
  # frequencies = {((a, b), delta): count for ((a, b), delta, count)}
  u = np.unique(data[:, -1])
  # z = itertools.chain.from_iterable([((a, b), delta, count) for (delta, count) in
  #                                    zip(*np.unique(axis=0, return_counts=True))]
  #                                   for (a, b) in itertools.product(u, u))
  l = data.shape[0]
  if l == 1:
    return data, d, False
  if sample_ratio < 1.0:
    #  index = index[np.random.randint(l ** 2, size=int((l ** 2) * sample_ratio))]
    index = np.random.randint(l, size=(int((l ** 2) * sample_ratio), 2))
  else:
    index = np.array(list(np.ndindex(l, l)))
  # print('index: ', index[:10])
  # can we somehow use a convolution (or similar transform) to compute this more efficiently?
  y = np.stack([np.concatenate([data[j, :dims] - data[i, :dims], [data[i, -1]], [data[j, -1]]]) for i, j in index if i != j], axis=0)
  # print(y)
  z, counts = np.unique(y, axis=0, return_counts=True)
  if counts.max() == 1:
    return data, d, False

  best = z[counts.argmax()]
  # print(z, counts)
  # print(best)
  # print(data, d)
  print(best, counts.max())
  rule = best[2:]
  left, right = rule
  delta = best[:dims]
  # print(rule, delta)
  # for row in data[data[:, -1] == left]:
  new_rule = np.concatenate([[np.max(d[:, 0]) + 1], rule, delta])
  merged_mask = np.full((l,), True)
  for i in range(l):
    if (row := data[i])[-1] == left and merged_mask[i]:
        # print(row)
      # print(np.hstack([data[:, -1, np.newaxis], data[:, :dims] - row[:dims]]))
      mask = np.logical_not(np.logical_and(data[:, -1] == right, (data[:, :dims] - row[:dims] == delta).all(axis=1)))
      mask_size = np.logical_not(mask).astype(int).sum()
      assert mask_size <= 1, mask
      if mask_size == 1:
        # data = data[mask]
        merged_mask = np.logical_and(merged_mask, mask)
        data[i, -1] = new_rule[0]
        # l = data.shape[0] # TODO
  data = data[merged_mask]
  d = np.append(d, new_rule[np.newaxis, ...], axis=0)
  # d = d[np.isin(d[:, 0], data[:, -1])]
  return data, d, True
```


```python
q = np.array([[[1, 2, 3], [4, 5, 6]], [[3, 2, 3], [7, 5, 6]]])
print(np.diff(q, axis=0))
print(q[0, 0, :])
```

    [[[2 0 0]
      [3 0 0]]]
    [1 2 3]



```python
def compress_array(data: np.ndarray, difference: bool = True, **kwargs) -> CompressedData:
  dims = len(data.shape)
  if difference:
    for i in  range(dims):
      data = np.diff(data, n=1, axis=i)
  print(data)
  plt.imshow(data)
  # root =
  dx, dy = data.shape
  xs, ys = np.mgrid[:dx, :dy]
  unique = np.unique(data, return_inverse=True)[1].reshape((dx, dy))
  vals, idx = np.unique(data, return_index=True)

  data = np.column_stack((xs.ravel(), ys.ravel(), unique.ravel()))
  d = np.hstack((np.arange(0, vals.size)[..., np.newaxis],
                                      data.flatten()[idx, np.newaxis],
                                      np.full((vals.size, dims + 1), -1))).astype(int)
  print(data, d)
  flag = True
  for i in range(50):
    data, d, flag = compression_round(data, d, dims, **kwargs)
    if not flag:
      break
  return data, d

with np.printoptions(threshold=5000):
  # print(compress_array(np.full((20, 20), 5), sample_ratio=0.2))
  print(compress_array(donut, difference=True))
  # print(compress_array(d3))

# TODO: prune unused rules, flatten deeply nested rules (?)
# TODO: handle overlap between matches
# TODO: use MCTS?
```

    [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0. -1.  0.  1.  0.  0. -1.  0.  1.  0.  0. -1.  0.  1.]
     [ 0.  1.  0. -1.  0.  0.  1.  0. -1.  0.  0.  1.  0. -1.]
     [-1.  0.  0.  0.  1. -1.  0.  0.  0.  1. -1.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 1.  0.  0.  0. -1.  1.  0.  0.  0. -1.  1.  0.  0.  0.]
     [ 0. -1.  0.  1.  0.  0. -1.  0.  1.  0.  0. -1.  0.  1.]
     [ 0.  1.  0. -1.  0.  0.  1.  0. -1.  0.  0.  1.  0. -1.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    [[ 0  0  1]
     [ 0  1  1]
     [ 0  2  1]
     [ 0  3  1]
     [ 0  4  1]
     [ 0  5  1]
     [ 0  6  1]
     [ 0  7  1]
     [ 0  8  1]
     [ 0  9  1]
     [ 0 10  1]
     [ 0 11  1]
     [ 0 12  1]
     [ 0 13  1]
     [ 1  0  1]
     [ 1  1  1]
     [ 1  2  1]
     [ 1  3  1]
     [ 1  4  1]
     [ 1  5  1]
     [ 1  6  1]
     [ 1  7  1]
     [ 1  8  1]
     [ 1  9  1]
     [ 1 10  1]
     [ 1 11  1]
     [ 1 12  1]
     [ 1 13  1]
     [ 2  0  1]
     [ 2  1  1]
     [ 2  2  1]
     [ 2  3  1]
     [ 2  4  1]
     [ 2  5  1]
     [ 2  6  1]
     [ 2  7  1]
     [ 2  8  1]
     [ 2  9  1]
     [ 2 10  1]
     [ 2 11  1]
     [ 2 12  1]
     [ 2 13  1]
     [ 3  0  1]
     [ 3  1  1]
     [ 3  2  1]
     [ 3  3  1]
     [ 3  4  1]
     [ 3  5  1]
     [ 3  6  1]
     [ 3  7  1]
     [ 3  8  1]
     [ 3  9  1]
     [ 3 10  1]
     [ 3 11  1]
     [ 3 12  1]
     [ 3 13  1]
     [ 4  0  1]
     [ 4  1  0]
     [ 4  2  1]
     [ 4  3  2]
     [ 4  4  1]
     [ 4  5  1]
     [ 4  6  0]
     [ 4  7  1]
     [ 4  8  2]
     [ 4  9  1]
     [ 4 10  1]
     [ 4 11  0]
     [ 4 12  1]
     [ 4 13  2]
     [ 5  0  1]
     [ 5  1  2]
     [ 5  2  1]
     [ 5  3  0]
     [ 5  4  1]
     [ 5  5  1]
     [ 5  6  2]
     [ 5  7  1]
     [ 5  8  0]
     [ 5  9  1]
     [ 5 10  1]
     [ 5 11  2]
     [ 5 12  1]
     [ 5 13  0]
     [ 6  0  0]
     [ 6  1  1]
     [ 6  2  1]
     [ 6  3  1]
     [ 6  4  2]
     [ 6  5  0]
     [ 6  6  1]
     [ 6  7  1]
     [ 6  8  1]
     [ 6  9  2]
     [ 6 10  0]
     [ 6 11  1]
     [ 6 12  1]
     [ 6 13  1]
     [ 7  0  1]
     [ 7  1  1]
     [ 7  2  1]
     [ 7  3  1]
     [ 7  4  1]
     [ 7  5  1]
     [ 7  6  1]
     [ 7  7  1]
     [ 7  8  1]
     [ 7  9  1]
     [ 7 10  1]
     [ 7 11  1]
     [ 7 12  1]
     [ 7 13  1]
     [ 8  0  2]
     [ 8  1  1]
     [ 8  2  1]
     [ 8  3  1]
     [ 8  4  0]
     [ 8  5  2]
     [ 8  6  1]
     [ 8  7  1]
     [ 8  8  1]
     [ 8  9  0]
     [ 8 10  2]
     [ 8 11  1]
     [ 8 12  1]
     [ 8 13  1]
     [ 9  0  1]
     [ 9  1  0]
     [ 9  2  1]
     [ 9  3  2]
     [ 9  4  1]
     [ 9  5  1]
     [ 9  6  0]
     [ 9  7  1]
     [ 9  8  2]
     [ 9  9  1]
     [ 9 10  1]
     [ 9 11  0]
     [ 9 12  1]
     [ 9 13  2]
     [10  0  1]
     [10  1  2]
     [10  2  1]
     [10  3  0]
     [10  4  1]
     [10  5  1]
     [10  6  2]
     [10  7  1]
     [10  8  0]
     [10  9  1]
     [10 10  1]
     [10 11  2]
     [10 12  1]
     [10 13  0]
     [11  0  1]
     [11  1  1]
     [11  2  1]
     [11  3  1]
     [11  4  1]
     [11  5  1]
     [11  6  1]
     [11  7  1]
     [11  8  1]
     [11  9  1]
     [11 10  1]
     [11 11  1]
     [11 12  1]
     [11 13  1]
     [12  0  1]
     [12  1  1]
     [12  2  1]
     [12  3  1]
     [12  4  1]
     [12  5  1]
     [12  6  1]
     [12  7  1]
     [12  8  1]
     [12  9  1]
     [12 10  1]
     [12 11  1]
     [12 12  1]
     [12 13  1]
     [13  0  1]
     [13  1  1]
     [13  2  1]
     [13  3  1]
     [13  4  1]
     [13  5  1]
     [13  6  1]
     [13  7  1]
     [13  8  1]
     [13  9  1]
     [13 10  1]
     [13 11  1]
     [13 12  1]
     [13 13  1]] [[ 0  1 -1 -1 -1]
     [ 1  0 -1 -1 -1]
     [ 2  1 -1 -1 -1]]



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-8-d5c9b9d87871> in <cell line: 26>()
         26 with np.printoptions(threshold=5000):
         27   # print(compress_array(np.full((20, 20), 5), sample_ratio=0.2))
    ---> 28   print(compress_array(donut, difference=True))
         29   # print(compress_array(d3))
         30 


    <ipython-input-8-d5c9b9d87871> in compress_array(data, difference, **kwargs)
         19   flag = True
         20   for i in range(50):
    ---> 21     data, d, flag = compression_round(data, d, dims, **kwargs)
         22     if not flag:
         23       break


    NameError: name 'compression_round' is not defined



    
![png](video-compression-old_files/video-compression-old_13_2.png)
    



```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from rich import print as rprint
from rich.pretty import pprint as pprint2
from pprint import pprint
from datasets import load_dataset
import itertools
from collections import Counter
import numba
import pickle
import zlib
import base64
# import gzip
```


```python
def make_donut(res: int = 25) -> np.ndarray:
  xx, yy = np.mgrid[:res, :res]
  r = res / 3
  r2 = 1
  k = 2
  circle = ((xx - res / 2) * 1.1) ** k + (yy - res / 2) ** k
  # donut = np.logical_and(circle < (r + r2) ** 2, circle > (r - r2) ** 2).astype(int)
  donut = (circle / 20).round() + 20 + np.random.normal(0, 0.3, (res, res)).round()
  donut = donut.astype(int)
  return donut

donut = make_donut()
plt.imshow(donut)

donut2 = np.tile(np.tile(donut, 2).T, 2).T
# plt.imshow(donut2)
```


    
![png](video-compression-old_files/video-compression-old_15_0.png)
    



```python
def generate_indices(shape: tuple) -> np.ndarray:
  if isinstance(shape, np.ndarray):
    shape = tuple(shape)
  return np.indices(shape).transpose(*range(1, (l := len(shape))+1), 0).reshape((np.array(shape).prod(), l))
# print(generate_indices((3, 3, 3))[1:] + 1)
```


```python
def segment(data: np.ndarray, w: Tuple[int, ...], ns: np.ndarray) -> np.ndarray:
  assert not (np.array(data.shape) % np.array(w)).any(), (data.shape, w)
  return sliding_window_view(data, w, axis=(0, 1))[tuple(slice(None, None, j) for j in w)].reshape((int(ns.prod()), *w))
l = 4
# print(z := segment(np.arange(l ** 2).reshape((l, l)), s := (2, 1), np.array([l, l]) // np.array(s)), z.shape)
```


```python
# CompressionDict = Tuple[np.ndarray, np.ndarray]
# CompressionDict = dict[int, np.ndarray]
CompressionDict = list[np.ndarray]
Shape = Tuple[int, ...]
CompressedData = Tuple[np.ndarray, CompressionDict]
Compressed = Tuple[CompressedData, list[Tuple[Shape, Shape, int]]]

def compression_round_2(data: np.ndarray, d: CompressionDict, dims: int, sample_ratio: float = 1.0,
                        window_size: int = 8) -> Tuple[CompressedData, Shape, Shape, int]:
  # print(data, d, dims)
  # assumption: all window sizes reduce size of compressed data (but not dictionary) by the same amount
  def compress_window(w: Tuple[int, ...] = (1, 2)) -> Tuple[CompressedData, Shape, int]:
    w_ = np.array(w)
    padding = w_ - (np.array(data.shape) % w_)
    dp = np.pad(data, [(0, i) for i in padding], mode='reflect')
    new_shape = np.array(dp.shape) // w_
    windows = segment(dp, w, new_shape)
    # k, v = d
    # print(w, windows[:20])
    values, idx = np.unique(windows, return_inverse=True, axis=0)
    return ((idx2 := idx + (km := len(d))).reshape(tuple(new_shape)),
          # d + [list(x) for x in values.reshape((values.shape[0], -1))]
          d + list(values)
          ), tuple(padding), values.shape[0]
  # ws = [tuple((np.eye(dims)[i] + 1).astype(int)) for i in range(dims)]
  # TODO: restrict to size of data
  ws = list(map(tuple, generate_indices(np.minimum(np.array(data.shape), np.full(dims, window_size)))[1:] + 1))
  ((data_, d_), p, n), w = min(zip(map(compress_window, ws), ws),
             #  key=lambda x: len(x[0][1]))
             key=lambda x: len(compress_bytes(x[0])))
  return (data_, d_), w, p, n

# Tuple[CompressedData, list[np.ndarray]]:
def compress_array_2(data: np.ndarray, difference: bool = True, **kwargs) -> Compressed:
  dims = len(data.shape)
  # TODO: pad (e.g., padding = v - (a.shape % v), (a.shape + padding) % v == <0, ...>)
  # TODO: early stopping
  # TODO: automatically test different shifts/offsets to better align patterns
    # (can also roll just rows/columns)
  # TODO: column permutations? other easily reversible transforms?
  # maybe: search for matching noise patterns that can be compressed to a single seed value
  # TODO: only store starting rule/pattern
  # TODO: use rule encodings that attempt to match actual values (so intra-round differencing is more effective) ?
  # TODO: denoising methods? (i.e., decompose into base signal and noise, compress separately, recombine)
  # TODO: pack metadata into single array for more efficient serialization
  # TODO: automatic refactoring of array code
  if difference:
    # TODO: re-include initial row so it is differenced along the other axis (so we don't need to include both in the metadata -- gains are amplified for e.g., 3D arrays with small cross-sections (I think))
    data = diff(data)
  # print(data)
  # rprint(compress_bytes(data))
  # plt.imshow(data); plt.show()
  # root =
  dx, dy = data.shape
  xs, ys = np.mgrid[:dx, :dy] # TODO

  u, idx = np.unique(data, return_inverse=True)
  data = idx.reshape(data.shape)
  d = list(u) # ?
  # d = dict(zip(d := np.unique(data), map(list, d.reshape((d.size, 1)))))
  q = np.inf
  ws = []
  for i in range(15):
    (data, d), w, p, n = compression_round_2(data, d, dims, **kwargs) # TODO: reindex
    # assert list(d.values())[-1].shape == w
    ws.append((w, p, (n,))) # TODO...
    # plt.imshow(data); plt.show()
    if (q2 := len(compress_bytes((data, d)))) >= q:
      break
    q = q2
  return d + [data], ws# + [(data.shape, (0, 0))]

def compress_bytes(data: Compressed) -> bytes:
  return zlib.compress(pickle.dumps(data), level=9)

def compress_2(data: np.ndarray, **kwargs) -> bytes:
  return compress_bytes(compress_array_2(data, **kwargs))

def pack_metadata(m: Compressed) -> np.ndarray:
  d, meta = m
  print(meta)
  meta_ = np.array([list(itertools.chain.from_iterable(x)) for x in meta]).flatten()
  dims = len(meta[0][0]) # TODO
  return np.concatenate([[dims, meta_.size], meta_, *(di.flatten() for di in d)], axis=0)

with np.printoptions(threshold=5000):
  # print(compress_array(np.full((20, 20), 5), sample_ratio=0.2))
  pprint(compress_array_2(donut, difference=True))
  rprint(base64.b64encode(compress_2(donut, difference=True)))
  # pprint(pack_metadata(compress_array_2(donut, difference=True)))
  # rprint(base64.b64encode(compress_bytes(pack_metadata(compress_array_2(donut, difference=True)))))
  # rprint(base64.b64encode(compress(donut, difference=False)))
  # print(compress_array(d3))
```

    ([-3,
      -2,
      -1,
      0,
      1,
      2,
      37,
      array([[1, 2]]),
      array([[1, 3]]),
      array([[1, 4]]),
      array([[1, 5]]),
      array([[2, 2]]),
      array([[2, 3]]),
      array([[2, 4]]),
      array([[2, 5]]),
      array([[3, 1]]),
      array([[3, 2]]),
      array([[3, 3]]),
      array([[3, 4]]),
      array([[3, 5]]),
      array([[4, 0]]),
      array([[4, 1]]),
      array([[4, 2]]),
      array([[4, 3]]),
      array([[4, 4]]),
      array([[4, 5]]),
      array([[5, 1]]),
      array([[5, 2]]),
      array([[5, 3]]),
      array([[6, 2]]),
      array([[12, 17, 17, 17, 17, 17, 17],
           [18, 12, 17, 17, 17, 17, 17],
           [16, 18, 13, 16, 17, 17, 17],
           [19, 15, 22, 22, 27, 22, 27],
           [15, 19, 13, 13,  9, 13,  9],
           [16, 23, 17, 17, 17, 17, 17],
           [25,  7, 23, 10, 17, 17, 17]]),
      array([[16, 28, 16, 23, 16, 16, 16],
           [16, 17, 18, 12, 18, 18, 18],
           [ 9, 14,  9, 16, 17, 17, 17],
           [28, 22, 21, 28, 17, 17, 17],
           [ 9, 14,  9, 16, 17, 17, 17],
           [28, 22, 21, 28, 17, 17, 17],
           [ 9, 14,  9, 16, 17, 17, 17]]),
      array([[17, 17, 17, 22, 17, 17, 17],
           [14, 16, 17, 18, 17, 17, 17],
           [15, 28, 17, 16, 17, 17, 17],
           [24,  9, 16, 18, 17, 17, 17],
           [16, 22, 23, 16, 23, 17, 23],
           [17, 17, 17, 18, 12, 23, 12],
           [23, 17, 17, 17, 17, 12, 17]]),
      array([[17, 18, 12, 17, 22, 17, 14],
           [17, 16, 23, 17, 13, 22, 17],
           [22, 17, 18, 12, 22, 18, 17],
           [18, 17, 16, 22, 18, 16, 17],
           [22, 17, 18, 17, 16, 18, 13],
           [18, 17, 16, 23, 17, 17, 26],
           [23, 17, 13, 17, 10, 17,  9]]),
      array([[21, 18, 18,  8, 18, 12, 22],
           [10, 16, 22, 18, 13, 16, 17],
           [16, 13, 18, 16, 27, 13, 17],
           [13, 22, 16, 18, 10, 16, 17],
           [17, 18, 12, 13, 16, 18, 17],
           [17, 16, 23, 22, 17, 17, 17],
           [17, 17, 17, 17, 17, 17, 21]]),
      array([[23, 17, 22, 13, 21, 22, 22],
           [23, 22, 17, 22, 17, 23, 13],
           [23, 13, 17, 22, 27,  9, 19],
           [28, 17, 17, 10, 10, 12, 20],
           [23, 13, 17, 22, 27,  9, 19],
           [28, 17, 17, 10, 10, 12, 20],
           [23, 13, 17, 22, 27,  9, 19]]),
      array([[23, 17, 24, 19, 18, 24, 18],
           [12, 23, 16, 22, 23, 17, 23],
           [22, 13, 14,  8, 17, 17, 17],
           [18, 16, 26, 18, 17, 17, 17],
           [ 9, 18,  9, 22, 17, 17, 17],
           [27, 12, 23, 17, 18, 13, 18],
           [12, 23, 12, 17, 16, 22, 16]]),
      array([[29, 11, 11, 16, 12, 17, 12],
           [12, 17, 17, 13, 17, 12, 23],
           [ 8, 17, 13, 22, 22, 23, 17],
           [12, 17, 22, 17, 13, 17, 12],
           [12, 17, 17, 13, 17, 16, 24],
           [12, 17, 17, 22, 16, 24, 11],
           [12, 17, 12, 23, 22, 17, 18]]),
      array([[37, 36],
           [34, 32],
           [33, 30],
           [35, 31]])],
     [((1, 2), (1, 1), (23,)), ((7, 7), (2, 1), (8,))])



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008000; text-decoration-color: #008000">b'eNrVWE1P20AQ9VdwHGxiu41NCy0thSqnnHvooWpujZRD76gKwZKRICCTHHLrhQCSbyz/pX+uTZ1kXcGY0ezQE0biycOb8cw+78zin9bdrxfa8joQ7</span>
<span style="color: #008000; text-decoration-color: #008000">bw1mpyeTzvDsyzpnE5OxseDLBtMRb52MRycDDJxK/LaklLg0Xh6niwsxvEncXN9Jb6Lds/M9c+i3+9/mxfX8ldPG4vDbv33fHWJWcFLzdTp1v9UTfOq</span>
<span style="color: #008000; text-decoration-color: #008000">SZPXPZNeNRlV0/4Dk567P7JkeDa6GGeT4bjIO7Vye3S0KvBW9LRL0dUPxaoMvfgxZiJ1brp++bR/jyjqSXfTvYVH+hHnm0y+xeTXFPkGM3+Dmb/BzN9</span>
<span style="color: #008000; text-decoration-color: #008000">g5l/moTP5BpNvMvkWk69abxlXY/J1Jt9g8k0m32LyVdenxqy3xqy3xqx3TSm+3bOX/K/XRd6upAZMDCVy/X3g7wE75icng9aUGAHcQu6bwN8D6CD3ZT</span>
<span style="color: #008000; text-decoration-color: #008000">4xs75XEm3g3yD8FKQqU9pGUvQJxKQLESyXYgNZGqyUbSBFC9gxv+fyPAWpMNeICL1BSIb5NZGSqVI2kdJDxV0cIa8g3DUxs5vEyD21+9ynS4U1tAiRK</span>
<span style="color: #008000; text-decoration-color: #008000">CBK9hD/iCg9QiQICQlCRPJIMQ5syKFinSW+JtYhQBqhoy5VC2lUdWIpG8SSYTPIR5bEBzPGI6SHS9wgdiesw1PcnVCiiDm7KGypSxUjb6EHQkVII6F2</span>
<span style="color: #008000; text-decoration-color: #008000">TQzixYgU8DjggOMANiMaAEspXj6z5z1Bqk0QIgR2rFdjMyFApPVAQ60rzggfNBxqVjnIsYbaHVvEDMIaJjbDsMbtq0v1RlLXAfrIo1zikO4h/BiRxCN</span>
<span style="color: #008000; text-decoration-color: #008000">2baA4Q73/zNMHr6RLHLcgf51Yr5iYnY9LZZX/h30pv/fsSdyV+E7ie4lvJX6QuHMvfnIg2mrfmGb4tyj9Ia8XX4or+Se7yrYfi2rQUeuLqMlMdP4Cjy</span>
<span style="color: #008000; text-decoration-color: #008000">QEBA=='</span>
</pre>




```python
# list(map(sum, [((1, 2), (3, 4))]))
list(range(5, -1, -1))
```




    [5, 4, 3, 2, 1, 0]




```python
def decompress_bytes(data: bytes) -> Compressed:
  return pickle.loads(zlib.decompress(data))

def decompress(data: bytes, difference=True) -> np.ndarray:
  d, meta = decompress_bytes(data)
  a = d[-1]
  # for w, p, n in reversed(meta + [((1, 1), (0, 0), (None,))]):
  for w, p, n in reversed(meta):
    b = np.empty(np.prod(a.shape), dtype=object)
    b[:] = [d[x] for x in a.flatten()] # awful hack
    a = np.block(b.reshape(a.shape).tolist())
    idx = tuple(slice(0, (-i if i > 0 else None)) for i in p)
    print(p, idx)
    a = a[idx]
  a = np.vectorize(lambda x: d[x])(a)
  dims = len(a.shape)
  print(dims)
  if difference:
    a = dediff(a)
  return a

# rprint(decompress_bytes(compress(donut)))
# plt.imshow(decompress(compress(donut)))
z = decompress(compress(donut, difference=True), difference=True)
plt.imshow(z)
# print(z)
```

    (2, 1) (slice(0, -2, None), slice(0, -1, None))
    (1, 1) (slice(0, -1, None), slice(0, -1, None))
    2





    <matplotlib.image.AxesImage at 0x7b93d0b4be80>




    
![png](video-compression-old_files/video-compression-old_20_2.png)
    



```python
def diff(data: np.ndarray) -> np.ndarray:
  for i in range(dims := len(data.shape)):
    data = np.concatenate([np.expand_dims(np.take(data, indices=0, axis=i), i), np.diff(data, n=1, axis=i)], axis=i)
  return data

def dediff(data: np.ndarray) -> np.ndarray:
  for i in range(len(data.shape)-1, -1, -1):
    data = data.cumsum(axis=i)
  return data
```


```python
print(compression_rate(donut, difference=True))
print(zlib_compression_rate(donut))
```

    9.25044404973357
    12.579710144927537



```python
def compression_rate(f: Callable[np.ndarray, bytes], data: np.ndarray, **kwargs) -> float:
  return len(data.dumps()) / len(f(data, **kwargs))

def zlib_compression_rate(data: np.ndarray) -> float:
  return len(s := data.dumps()) / len(zlib.compress(s, level=9))
```


```python
a = np.random.randint(0, 2, (30,) * 3)
print(compression_rate(a))
print(zlib_compression_rate(a)) # performs worse than zlib if geometrically unstructured
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[7], line 2
          1 a = np.random.randint(0, 2, (30,) * 3)
    ----> 2 print(compression_rate(a))
          3 print(zlib_compression_rate(a)) # performs worse than zlib if geometrically unstructured


    Cell In[6], line 2, in compression_rate(data, **kwargs)
          1 def compression_rate(data: np.ndarray, **kwargs) -> float:
    ----> 2   return len(data.dumps()) / len(compress_3(data, **kwargs))


    NameError: name 'compress_3' is not defined



```python
# TODO: test out-of-the-box neural compressors
# n-dimensional Hilbert curve?
# test mixed window sizes for curve segmentation
# can also probably optimize by creating a function to map indices from 1d to 2d, though this may be more complex to implement
```


```python
def segment_ragged(a: np.ndarray, r: int = 2) -> list[np.ndarray]:
  out = [a]
  dims = len(a.shape)
  if isinstance(r, int):
    r = [r] * dims
  for i in range(dims):
    out = list(itertools.chain.from_iterable(np.array_split(x, r[i], axis=i) for x in out))
  return out

# a2 = np.arange(343).reshape((7, 7, 7))
a2 = np.arange(121).reshape((11, 11))
print(segment_ragged(a2))
```

    [array([[ 0,  1,  2,  3,  4,  5],
           [11, 12, 13, 14, 15, 16],
           [22, 23, 24, 25, 26, 27],
           [33, 34, 35, 36, 37, 38],
           [44, 45, 46, 47, 48, 49],
           [55, 56, 57, 58, 59, 60]]), array([[ 6,  7,  8,  9, 10],
           [17, 18, 19, 20, 21],
           [28, 29, 30, 31, 32],
           [39, 40, 41, 42, 43],
           [50, 51, 52, 53, 54],
           [61, 62, 63, 64, 65]]), array([[ 66,  67,  68,  69,  70,  71],
           [ 77,  78,  79,  80,  81,  82],
           [ 88,  89,  90,  91,  92,  93],
           [ 99, 100, 101, 102, 103, 104],
           [110, 111, 112, 113, 114, 115]]), array([[ 72,  73,  74,  75,  76],
           [ 83,  84,  85,  86,  87],
           [ 94,  95,  96,  97,  98],
           [105, 106, 107, 108, 109],
           [116, 117, 118, 119, 120]])]



```python
def unroll(a: np.ndarray, r: int = 2) -> np.ndarray:
  if a.size <= 1: return a.flatten()
  return np.concatenate([unroll(x, r) for x in segment_ragged(a, r)], axis=0)
print(unroll(a2))

# def reconstitute(a: np.ndarray, r: int = 2) -> np.ndarray:
```

    [  0   1  11  12   2  13  22  23  24   3   4  14  15   5  16  25  26  27
      33  34  44  45  35  46  55  56  57  36  37  47  48  38  49  58  59  60
       6   7  17  18   8  19  28  29  30   9  20  10  21  31  32  39  40  50
      51  41  52  61  62  63  42  53  43  54  64  65  66  67  77  78  68  79
      88  89  90  69  70  80  81  71  82  91  92  93  99 100 101 110 111 112
     102 103 104 113 114 115  72  73  83  84  74  85  94  95  96  75  86  76
      87  97  98 105 106 107 116 117 118 108 109 119 120]



```python
def unroll_2(a):

```


```python
unroll(a2)[:60]
```




    array([ 0,  1, 11, 12,  2, 13, 22, 23, 24,  3,  4, 14, 15,  5, 16, 25, 26,
           27, 33, 34, 44, 45, 35, 46, 55, 56, 57, 36, 37, 47, 48, 38, 49, 58,
           59, 60,  6,  7, 17, 18,  8, 19, 28, 29, 30,  9, 20, 10, 21, 31, 32,
           39, 40, 50, 51, 41, 52, 61, 62, 63])




```python
unroll(donut)[:100]
```




    array([37, 35, 35, 35, 35, 33, 34, 33, 34, 33, 33, 32, 32, 31, 31, 31, 33,
           32, 32, 31, 32, 30, 30, 29, 29, 28, 29, 28, 32, 31, 30, 30, 30, 29,
           29, 28, 31, 29, 28, 27, 28, 27, 27, 26, 26, 26, 26, 25, 25, 31, 30,
           30, 28, 30, 29, 28, 28, 27, 26, 27, 26, 30, 30, 28, 28, 29, 28, 27,
           27, 26, 26, 27, 25, 26, 25, 25, 24, 24, 24, 24, 24, 23, 25, 24, 24,
           24, 24, 23, 23, 23, 23, 30, 28, 29, 28, 27, 26, 27, 26, 29])




```python
def compress_3(data: np.ndarray) -> bytes:
  # return zlib.compress(unroll(diff(data)).dumps(), level=9)
  # return zlib.compress(diff(unroll(data)).dumps(), level=9)
  # return zlib.compress(diff(unroll(diff(data))).dumps(), level=9)
  return zlib.compress(diff(data).tobytes(), level=9)
```


```python
d = make_donut(100)
# d = np.random.randint(0, 5, (20,) * 3)
print(compression_rate(compress_2, d))
print(compression_rate(compress_3, d))
print(zlib_compression_rate(d))
```

    17.09961295579548
    23.28488210818308
    9.16597510373444



```python
# ds = load_dataset('commaai/commavq', num_proc=num_proc, data_dir='~/dataset')
# ds = load_dataset('commaai/commavq', streaming=True)
ds = load_dataset('commaai/commavq', num_proc=16)
# it is nice to work on something where overfitting is encouraged, for a change.
```


```python
d = ds['0']
```


```python
list(d)
```




    [{'path': '3b41c0fa8959aea6c118e5714f412a2e_13.npy'},
     {'path': '514520784bd7b5e0cfacddce40092428_7.npy'},
     {'path': 'aa4887f49431bcb803c00b93acc3a018_10.npy'},
     {'path': 'db69bc3bf1827a76b44cbf6114bf1581_31.npy'},
     {'path': '2c227b3b141bd45319c5a48d020954f2_13.npy'},
     {'path': '774403dc5aeaf1323fc671832735a1e3_214.npy'},
     {'path': '57de94288a86a998b7e45ec5c991965e_1.npy'},
     {'path': '0e7f939462b833a5cd485aa9ff8fea8f_16.npy'},
     {'path': '1bf30722776c8c9b4c389f7b60503122_8.npy'},
     {'path': '206461e1ace4da5140da1e1bbd377297_24.npy'},
     {'path': '2fd3114472f69b87f6473b52046d4b4f_27.npy'},
     {'path': 'b660c0dc3c04c17cbba56d253d7a9fc9_4.npy'},
     {'path': 'f0bdd51d73e6e35b3f30d2a39e7de5f2_10.npy'},
     {'path': '82f245f8a61f344dee69ec669869a2c9_17.npy'},
     {'path': '306a843bf0d5e8432076f88631fcb871_6.npy'},
     {'path': 'fa7ee9eb8d48e55b9db9bb763a969013_10.npy'},
     {'path': '3443b96d21ac877daa46c34bbddaa09b_6.npy'},
     {'path': 'd08232e8180eb11dc7f80e7be2d4cd4a_2.npy'},
     {'path': '69cd1dfa915bdbb5f13f32e72a1e4ca9_21.npy'},
     {'path': '131aa5482de6d37cb301b2cfb210dda8_19.npy'},
     {'path': '5ddd4a8603dc25c5a47b7a72972edb3c_12.npy'},
     {'path': '966fdf10c88098369920248904116d7a_169.npy'},
     {'path': 'b0b0a7a72f8ab63d3aead0980f5d5cf5_6.npy'},
     {'path': '3fbec25841065d6ea8f2973483c2e860_3.npy'},
     {'path': '6b76985d8d6da88416b133361c681f85_63.npy'},
     {'path': '8a821fc82edbcbcf2c7892a2e09ad34c_25.npy'},
     {'path': '7bc943af724324316024dac17ada24d1_64.npy'},
     {'path': '60413816432f5ebdeaccfeeac151dfcb_2.npy'},
     {'path': 'e999b87a78ced279bdf887edafdb5f5d_6.npy'},
     {'path': '37e892c5c27e8dbc239ab307619a8a9a_3.npy'},
     {'path': 'b8f746af75cbd481812e43260ce69e46_5.npy'},
     {'path': 'a90f35237e9b6ebb903f012ed3cb94c5_76.npy'},
     {'path': '7664ac276248e10bbbad6b46873cc86b_63.npy'},
     {'path': '23617ea465e8c4a014a360fe003c9688_5.npy'},
     {'path': '6586ec95e99d635a5e7091aed3c728d4_0.npy'},
     {'path': '9f63b8679654ff002ee411cd5e7f3a0d_23.npy'},
     {'path': '038c874f2a5c51f36171e0d504cc04cc_13.npy'},
     {'path': 'cf11ad7815a49d438cecbc0ba27ff31b_48.npy'},
     {'path': 'd2c06243589bf9b98a54341279b081bb_6.npy'},
     {'path': 'b98b520656473dabad0f1b9b624855ff_11.npy'},
     {'path': 'e2c09cdb57045857bcfbb8836ca58b67_14.npy'},
     {'path': '6baa4e5af260504078d0005e84884b45_8.npy'},
     {'path': 'dc0f4795808e22089b36f92ccee06e91_6.npy'},
     {'path': '7f2572917e6f29bcecaa96acd20a98c4_6.npy'},
     {'path': 'b2d977173d0c502f0e19ce8c9660d42b_6.npy'},
     {'path': '50722362feea7a344aa17bbac70212bb_4.npy'},
     {'path': '6f71500cc2686174f45312dd72ee5272_45.npy'},
     {'path': '9f0fe4e90a35a0773a80e4a6c1c80d89_20.npy'},
     {'path': '3f1402d02a298c00b2065b326afc30e9_3.npy'},
     {'path': '7d616267c4aacff0f41feb1065f0df1a_83.npy'},
     {'path': 'ffb1035fe1454ab3ebfccad8124c327f_15.npy'},
     {'path': 'c406c512c66e4ac21b548b6f958cce5a_6.npy'},
     {'path': '34973a46a6c0bcd1d576c3c680802953_12.npy'},
     {'path': 'ccc0d965a41af37a1bff9a3788dac4c3_26.npy'},
     {'path': '023ae07fda627a30d0f10ac86735692a_65.npy'},
     {'path': '4a090106631f20de92365b293ccbc5f4_13.npy'},
     {'path': '8fd2816d2d925d0d6b6ffadbd6602147_1.npy'},
     {'path': '6796d15e53b11a231e451cd97fc0e8dc_59.npy'},
     {'path': '8230d06e4b6ab5ea66ac34ca21209a4c_53.npy'},
     {'path': '289cc7b78e9a2dff8a331db293c8ddfd_43.npy'},
     {'path': '54e914bf7fdf798fbbe6708cd6a0498a_5.npy'},
     {'path': 'd5f85000374bc0ecfb35e5b854f44e65_5.npy'},
     {'path': 'a53ca51260369096abac08b3721e30bd_0.npy'},
     {'path': '0ad8fd36693721f2794959688b2f51c9_55.npy'},
     {'path': '3ad1cf1676796a8edadef70810422129_3.npy'},
     {'path': 'bd631298a58d89665f13177be4dc67ed_7.npy'},
     {'path': '88fee5ab2b15ed8f460fb91012fc2044_19.npy'},
     {'path': 'a99c908a3cdcbb91bc1eaa3d889c0ef8_29.npy'},
     {'path': '1c1a8567818977632e1c903766db3b1a_7.npy'},
     {'path': 'c140741008cffa59309ee0924a819571_0.npy'},
     {'path': 'c1599c49f5ba277664fd66962c436bac_42.npy'},
     {'path': 'a4ad97e9cabe78f4941973abcddc0778_4.npy'},
     {'path': '48966d3272fbc8c9a8bd8ee21a83d321_14.npy'},
     {'path': '10840117907bc6b825b47bed360e4619_22.npy'},
     {'path': 'b4ef5ce20fe2df0fee14f354acd53d27_24.npy'},
     {'path': '04feb3a5e341965207fb84e2a8d73657_41.npy'},
     {'path': '1ff78a5226d679869ebe372ca979e2ab_7.npy'},
     {'path': '3239d15c2c28c2396611f7721e3d3280_30.npy'},
     {'path': 'd6ea0fe460fd91a01868f5928f45a682_2.npy'},
     {'path': 'c12b9367b20e4b559f26749927767d0e_2.npy'},
     {'path': 'f4efa917a2a2e9d0c5e4bc12a31ec37b_10.npy'},
     {'path': '987d025555b4fd7c5902654297982242_3.npy'},
     {'path': '38df9fdc8f316fd89c799a432ef3434e_6.npy'},
     {'path': 'bba892d5ba82269fc247606ee1df33ff_2.npy'},
     {'path': '239b80dd6d4423baaf1d56ad8dcb87c2_4.npy'},
     {'path': '6541f2fde1ec35effefa030d62a85b9c_11.npy'},
     {'path': 'cfc0b9b6e49b263618f73187c88ccdf6_3.npy'},
     {'path': 'd4c0558467604eb9da826343785014d4_22.npy'},
     {'path': 'df48107ddba8a08bf5ecd98bbd964734_7.npy'},
     {'path': '64f83f6d95303d2bb449ea8d9e5301cb_94.npy'},
     {'path': '044646730f2e07d9c65ad5f0510573ce_7.npy'},
     {'path': '3dac1d280aff8837d422d4aab2ca42df_3.npy'},
     {'path': '613afd8691a4537122af5a6eda8b726a_8.npy'},
     {'path': 'a6af26234b8f1cd22354ab06ab1d9de7_40.npy'},
     {'path': 'dffb9dfe98b378b61db8506e04cdd6d9_68.npy'},
     {'path': '3a91c9874707e5a664c85c5d9bed46e9_10.npy'},
     {'path': 'b8cd65f45a10e6c4f6b341fc79a322e1_2.npy'},
     {'path': 'cf762dbf79ac69598a7c771590405eae_8.npy'},
     {'path': 'b44ddb9dddaa36a5c00588679d46a7e7_21.npy'},
     {'path': '55c41db0b7e41ecd4a7aa3a39abdc873_3.npy'},
     {'path': '6e44258944447c8e0ffa716c2d44d50e_6.npy'},
     {'path': '540f374c1f6b644eedc9d7662f77b0d0_28.npy'},
     {'path': '8dcb453836e54d9776831a5105052448_2.npy'},
     {'path': '26434c4285a7434e66d1ccdaa2197a3e_15.npy'},
     {'path': '5c2b6db074b06b1f71e5d9b3c9b4f34e_1.npy'},
     {'path': '921aac0f65278c8bd67e5a48806f58f2_117.npy'},
     {'path': '5809aa3f06b8a64fdc3a6ed581fc6372_18.npy'},
     {'path': '0741835b76f9d927582a24a771d7feee_7.npy'},
     {'path': '264f6d497d7dd49ab9c80b8adf39c2fe_43.npy'},
     {'path': '63015c6fa614af1f2ae833c254fb0527_2.npy'},
     {'path': '70d51c5ced1a4f0f4450a1436004ca35_9.npy'},
     {'path': '3187f575c5cf9db720a2bff27e1a424b_51.npy'},
     {'path': '88b8450f168e68b8b4338460b2239f37_1.npy'},
     {'path': 'a699aaf12aefbc3e99b3210615c2ecf9_12.npy'},
     {'path': '4df0e106e42dc6cdbb937706ee6a8f67_21.npy'},
     {'path': '5eefa2f23b66eebf1851fd9d04ca6c40_6.npy'},
     {'path': '3084e37a366d6b6267db3b51af7fbf3e_6.npy'},
     {'path': 'b95ac5816c68962153638139df0f3dec_7.npy'},
     {'path': '5057c0b2e7a416e1e17ed6717fdb328e_3.npy'},
     {'path': '8fc9c514c2585603bea13eed2983ce16_21.npy'},
     {'path': '79af463c5b2db36818bf3349c2d099ef_80.npy'},
     {'path': '3afd156078ac47c073020534e20a6ab8_28.npy'},
     {'path': 'baabf327abcf1205914ced3e5c03bbc1_120.npy'},
     {'path': 'aa07b247a947bd2b722dfd3358a214d8_8.npy'},
     {'path': '8ac5dbdb7616de31ee217fb6ca0feb25_21.npy'},
     {'path': '71a00bba1cc84060f0382ced6b44a705_36.npy'},
     {'path': 'abe7e9b2bc841ab278e9b9afd3848a57_11.npy'},
     {'path': 'abe6fd2ec221fb98648c08b372533218_4.npy'},
     {'path': 'a55d8267bae33fcebdf3cc487709de31_10.npy'},
     {'path': '48876e966e232a50eb2a021a9cd55800_6.npy'},
     {'path': '9126e89847d1619e65ed20d24564d6f8_74.npy'},
     {'path': '8367937d374fc295c3d1402c1fdb9975_24.npy'},
     {'path': '5c0392660092399eac6d6aba182f2b26_29.npy'},
     {'path': 'e55d2e265ea288416616422d49e6ef08_25.npy'},
     {'path': '9acf349d48f134e29f69b1dfe462e9f8_21.npy'},
     {'path': '5d0f85253fe26c111d40ed2b82906a15_9.npy'},
     {'path': 'c1870edd1f8e25308b184b9cdb633b7a_20.npy'},
     {'path': 'ab32fd91ab6f68cfb001983ca79d1492_15.npy'},
     {'path': '2344d7950a1915f2d1614009119161c0_8.npy'},
     {'path': 'b14fc3d79a5b40e1f67f7678373b0753_17.npy'},
     {'path': 'bec88a3cf95a18965b10b89d010ee0bd_1.npy'},
     {'path': '1d2e87fd5b3b9d7fa9dcf15e3d66685a_11.npy'},
     {'path': '663967e0fec9ede1a27db5542ccd6fc8_2.npy'},
     {'path': '08a44d5289053a600a89a496df17d6b6_10.npy'},
     {'path': 'f2638ddd688cb01527a2bdf6ac7c59c4_10.npy'},
     {'path': 'fe5b1d87bccb837b5d7a458c5559caa1_16.npy'},
     {'path': '98e7a1771a8b1e17037bfc3a134badfe_7.npy'},
     {'path': '1e26a5f728958f4b593e4eb96ab3ade8_3.npy'},
     {'path': 'e143910b0080122a68a9d112c67cd83e_5.npy'},
     {'path': 'c2135a91eb23b1479f190e2b498d4bbc_2.npy'},
     {'path': 'd8d2206ac03e6ae94e56e46d05867330_50.npy'},
     {'path': '90d4645e6b0cbc22651fc8bf02d3448c_7.npy'},
     {'path': '61205d4084c524e89087296863444cd4_5.npy'},
     {'path': '87b5a38efc3db2b1dc6ec102df53f4db_13.npy'},
     {'path': '197fd74b77530e1b5b43e0392cab47df_85.npy'},
     {'path': '64f83f6d95303d2bb449ea8d9e5301cb_45.npy'},
     {'path': '758b8df68fd483c8ad6c5d96aeb3be76_23.npy'},
     {'path': 'b659bd73ac20471e6eadddc782db0965_17.npy'},
     {'path': '10a7788a480642c968f98b3f39e487a6_14.npy'},
     {'path': 'a00399b6fd1e0322c83be1e601bc53b2_77.npy'},
     {'path': '4c278d06a36cf586ced1a188a58d26f8_20.npy'},
     {'path': '3ee650af015cc34e7b55542963fa21d3_3.npy'},
     {'path': 'f054c3b69eb5e2746ba210d042936af5_4.npy'},
     {'path': 'ebcdd45fb4ac21fb6b8bc080fe38cf8a_7.npy'},
     {'path': 'fa01dce0a82104fd671f22460a03cdb7_18.npy'},
     {'path': 'd9240bb7fcfe22725901305e708a1931_17.npy'},
     {'path': 'f3210f65970859a34a3d0f0a2d39ed24_43.npy'},
     {'path': '0cd24484b1d85440deb36b5f29db13b3_2.npy'},
     {'path': '1e3bba0a44df85cc9c5172234908dec5_10.npy'},
     {'path': '0efd785f17ec6a269865c2507a2eb171_19.npy'},
     {'path': 'e8a510470931e98e2bcaa26f9830a4f9_9.npy'},
     {'path': '5dfd548303661560f9fa1559ac80f4c3_3.npy'},
     {'path': '63a13081eb826ed8ff1b38948b6f562f_16.npy'},
     {'path': '7cd76d02d0e143b1a7672c0e281b6dd1_5.npy'},
     {'path': '855e04b1f0ed8552445c9d0366903014_21.npy'},
     {'path': 'd809c6a631834f741f29e382d8824365_47.npy'},
     {'path': '7d616267c4aacff0f41feb1065f0df1a_52.npy'},
     {'path': '2445cd0349e815f0171dd5718d4b8bd0_10.npy'},
     {'path': 'df8706e274b985bb75ddfcef92b48194_2.npy'},
     {'path': '35230e9867021d08ace51b76e99bac9c_3.npy'},
     {'path': '6ddce77c7c228d54f7da521385014c3b_3.npy'},
     {'path': '02121f472831df61c87efd911158d9b2_51.npy'},
     {'path': '985fb7cdfc2f4a82c947757d823d32de_132.npy'},
     {'path': '00148e277029edef492f36a536304d7f_15.npy'},
     {'path': '19fe7663d1fa829f8cc25f5204db209a_4.npy'},
     {'path': 'a507b295c2bd31baae1b6a4d68f23420_8.npy'},
     {'path': 'd0c5de3c2f5bc3607e74a42bcde0e431_11.npy'},
     {'path': '4d5a02a1d91446ffe94796a497c46999_73.npy'},
     {'path': 'd346e9ad1a50b981d1469dea4b4e820f_45.npy'},
     {'path': '6365c27a27b1386a76e14c7ed77f825e_4.npy'},
     {'path': 'd046d3f50bed259fa9710d5e7ed783e0_54.npy'},
     {'path': '6dc062c00e34648f1af3a26d26f7a038_5.npy'},
     {'path': '290670cb0e1eacfd4d7b3ef509833ecf_14.npy'},
     {'path': '18048b4d61856952ef5f92a15fbef1da_12.npy'},
     {'path': 'ec4d0de7316d3da760dc18100c2cb154_7.npy'},
     {'path': '9d90aa72638dfc882fa65d13e344dfea_7.npy'},
     {'path': '2f7d05cca95ef832a3e52624a6a20904_10.npy'},
     {'path': '74f72b0e697573bf24d16b4d8e542657_29.npy'},
     {'path': 'd4cc2776251a7387f3c76b562ff54d01_27.npy'},
     {'path': 'acbd95da09dc8b060b8e9935414da9a0_82.npy'},
     {'path': 'f6b05e5b6f3eece966cb4775a7f79bc9_61.npy'},
     {'path': 'ffbc3f880eea75edfb5572efc1d67064_6.npy'},
     {'path': '491628d0774e60f7eea8dcecbb808fde_8.npy'},
     {'path': 'b4f6a2cb767044b14913b2c3e99b78ef_176.npy'},
     {'path': '4ed1d18236740fc38ad8b03f1d12fa84_15.npy'},
     {'path': 'f3604fb1c17183b248567199c625d5cc_35.npy'},
     {'path': '7011b46871d2b5de50bfa807833ff4b1_5.npy'},
     {'path': '049b3a1d62cf14ff6d93a8337dc07918_30.npy'},
     {'path': '13a62203fa3919f43a4f87f8907d78df_54.npy'},
     {'path': '45d389dfa53c9e8e1e7154fe1fd2ba71_15.npy'},
     {'path': 'ec061652bd4200241820473223de9dc9_74.npy'},
     {'path': '5f11771cf2d92ad06d2eab97b27ee23f_18.npy'},
     {'path': '57c8b8ccc977baa86f705c00d087d832_268.npy'},
     {'path': '7f55a8197ac2de555bf49e6998cdb4b2_55.npy'},
     {'path': '67ea6c4306540ea5e6026c11ae487e9a_5.npy'},
     {'path': '7b2914e5ead13d3606bf548e08036e84_52.npy'},
     {'path': 'b638ff6f94cbdda73b0813cfddba4b70_4.npy'},
     {'path': 'c321fb12d77ebd82fae6b64c0c98c668_22.npy'},
     {'path': '566ad2e4fb23042dfce8180749a6424a_7.npy'},
     {'path': '31d2b7d6b16c1af27d59dc340c82efdf_6.npy'},
     {'path': 'dc6e0725e66e209ebfdc5854aeb103de_26.npy'},
     {'path': '8f283fe8c2e4a95ea3f3ad011b431079_7.npy'},
     {'path': '70e19a6c89c40b5e38a25d4235b781b0_21.npy'},
     {'path': '330de1d5e48cc7cfb425a90aaa5038f0_7.npy'},
     {'path': 'f3db5e755dacc2f048428d4fa58cc3ee_10.npy'},
     {'path': '7f9e5a282e5a65cee123ab6c6be00ee7_9.npy'},
     {'path': '72fca029c3259f7f617b6d24a78f682d_8.npy'},
     {'path': '19b5e4d1819b1217dff7e6c31fbf277f_4.npy'},
     {'path': '2dfe621ace090e5e50c4fad74097481e_2.npy'},
     {'path': '4ab47db2ae0e17663f0bca7a9ba3ce27_1.npy'},
     {'path': '9d6c1fdc8fa6897835fad2a101f50104_3.npy'},
     {'path': '3bc9c40dfa1cac3c2b250e66010ef99b_29.npy'},
     {'path': 'd890a243357591e57946d03805ef1b6e_13.npy'},
     {'path': '4181864f083537b01a5db9e8b776df4e_57.npy'},
     {'path': '98793f1495487686a006bad10cff59e9_15.npy'},
     {'path': '740980307b45c0045c3f34b321e6f3b1_5.npy'},
     {'path': '3e841a6792d4175b77972404b4b190b2_9.npy'},
     {'path': '8c52c8d659ff5f41ba8cdd26dfcc4311_15.npy'},
     {'path': 'd6c7bef4a1fb01b0477ee08054f79ead_19.npy'},
     {'path': 'd8a96fa7272b4e31a5c756fe64a778b5_27.npy'},
     {'path': '11226021c75162812e76d95bb239e471_7.npy'},
     {'path': '245ee5b655b68c599f6b91a1a2e44010_70.npy'},
     {'path': 'c233eb2ec15e77e5f1dbf3693d38c197_11.npy'},
     {'path': '261c32b3e264439be6233f679d10e3db_18.npy'},
     {'path': '15bcb09f2814344d0a80716ff1f95dc1_10.npy'},
     {'path': '0829ac17a51892ad8ec30767a3d36c03_40.npy'},
     {'path': '5327f47c8e39e2cffd4c5ffc36c8d004_34.npy'},
     {'path': '7d13c307b87d71eec014e400439f6b14_3.npy'},
     {'path': '0c3915bcd0ee8cde99e228197a3aca8e_12.npy'},
     {'path': '14400daae73bb74d2f756394c1ee4a48_8.npy'},
     {'path': 'fff132b2764ea46ba24b9043559c8ae7_2.npy'},
     {'path': '5dc0479e2d7581dd394ac6649b64e5cc_46.npy'},
     {'path': 'a409ffdd8dfc7f94d393818c9d76dd61_11.npy'},
     {'path': 'b9ee57c0a909c54ccce1358d82c2cce4_11.npy'},
     {'path': 'ab333f4e52073f67bb0fbb9cce691f3c_7.npy'},
     {'path': '0b25d71368f428c91ef30de5e9677294_19.npy'},
     {'path': '519af903f91d06fb35c4676bfc770450_128.npy'},
     {'path': '8fa8d4255099a49d52a28372bab4ad14_15.npy'},
     {'path': '32be551db887bb0bda1cfa31e5f5f363_13.npy'},
     {'path': 'f2f1c320410f592132bcdb35bd90d198_22.npy'},
     {'path': '310b3a7905bad2bb972566024f998ef8_5.npy'},
     {'path': '998073b2e655d6f87a63a467e212996e_28.npy'},
     {'path': '520e89fad032db1f4858e16078252401_16.npy'},
     {'path': '7fb2683c7dbde22e98362fa3970c63bb_1.npy'},
     {'path': 'bc86590150d95b00932b3825c9466643_22.npy'},
     {'path': '8a2329775e69da743230f1c1dc6639c0_15.npy'},
     {'path': '0337d4c8fbcce22a0bc02ede4058bf06_30.npy'},
     {'path': 'c4eaa10745386a93b02b6f3ab59079f6_12.npy'},
     {'path': 'd0a1fdae9edba971bb70f3a810935ef2_6.npy'},
     {'path': 'e283053cd3d6ab022ca3d30bfef98b71_59.npy'},
     {'path': 'bb257808eaa7a7ba7a66c4e842345333_18.npy'},
     {'path': '642201c1e2eb30a1c3020985847fa172_30.npy'},
     {'path': 'ad37c9981a069b2391249f648123d5ba_25.npy'},
     {'path': '017176550d61d1781b12745999daa9f7_5.npy'},
     {'path': '435cd1491f439aaf791faffb15f7d66f_3.npy'},
     {'path': '20289afa2144903182e9714de119dcf6_6.npy'},
     {'path': '3698014def14f94d939406ecef43fd42_15.npy'},
     {'path': 'b500ae17c8b6c0f921d4e8000dc7fb98_21.npy'},
     {'path': '6fcae7129c03f4cfd8608613aaef4e1a_3.npy'},
     {'path': '4a8bf3ee685e7ba36c44112513d8a680_5.npy'},
     {'path': 'dc8f8a2127f9540256537d17482bb46a_1.npy'},
     {'path': '9169a32aaaeea34ca3a1aa4a350dbee4_24.npy'},
     {'path': '4a32f4c5b02e216e277c8471ebcf5616_49.npy'},
     {'path': '09145b579bbe1e1cae22a71fd6f56287_36.npy'},
     {'path': 'f7515877a2021f617fe7840162314f81_7.npy'},
     {'path': '1ab60175263231ee0461fc9b98ba7590_20.npy'},
     {'path': '064c96ced7061a354e88b928d8371861_7.npy'},
     {'path': '269cf2cf319a4da478dd45401568b795_18.npy'},
     {'path': 'ab9ac4c4309d76e0db5fa29982bbce94_12.npy'},
     {'path': '07d9c07dd3e93f83607169a82d9e8680_2.npy'},
     {'path': '64ed57d750c075b8055b46fe54df4afb_10.npy'},
     {'path': '7b10ad3f4579e0771ca3aba7b07720c5_16.npy'},
     {'path': 'f4a82611afff39e78612da05d10450bb_8.npy'},
     {'path': '00dd331938ba46ef377a4c5d063f75b8_49.npy'},
     {'path': '7cd039cc87159e42e5f42ff67072ae35_21.npy'},
     {'path': '18eb61caa1d78260cb253ed8725220ce_5.npy'},
     {'path': '81abd586bda69c5df139fa07b969f24d_6.npy'},
     {'path': '20dbb573312d29491dc323d9f8bcab30_0.npy'},
     {'path': 'a9c5da71646fd1f0bddb8fa5969e9089_31.npy'},
     {'path': 'a96db4d3af3f3c9e897fe701ab55842e_14.npy'},
     {'path': 'bfe0a47df3a003d0bca8af12e0b59fe5_57.npy'},
     {'path': '371ac1936ea62b1ac25806a89f362b4c_30.npy'},
     {'path': 'ed5de1886cde75527c808369d5ce91c1_40.npy'},
     {'path': '76b384d4887da607b5e474e7aaa333a1_4.npy'},
     {'path': '8a547b75ff8454541bc8b6ed49a09fb8_3.npy'},
     {'path': '0634ce372cf7f1639cf4fc237d1deaa8_15.npy'},
     {'path': '09f2bd7b9dfb6c9622b8b42783ff24e7_19.npy'},
     {'path': 'c158b89e30696a772d19b504fea71c16_15.npy'},
     {'path': '5322a7599481b600a00697bf599f724b_24.npy'},
     {'path': '419c890f3817ce2ccc57891bf451ceaf_1.npy'},
     {'path': '552446deba780dd660d8febb58e9fbfb_3.npy'},
     {'path': 'e12d8d0310ba53d3fcc3791a449bc239_27.npy'},
     {'path': 'ff043e3d92719db3bfd40df1a1412234_2.npy'},
     {'path': 'd571f224e2a6b3e962e6979d8894ef6a_52.npy'},
     {'path': '2254689f8d1ca6788ff55f9fe6484538_8.npy'},
     {'path': 'd588743327997d6aa768d1ec0837b60e_20.npy'},
     {'path': 'd5dc07ddecda817187f5797506d366f4_20.npy'},
     {'path': '02be35589bc829e3a8438b3cc52c7cd5_42.npy'},
     {'path': '5db3937fb6a6c5b02499a3345c86486f_2.npy'},
     {'path': 'c03d325cb8ccbb4bdfe433a67f512b38_0.npy'},
     {'path': '3c1e77b8264fea30cdb6e2587f749a1e_72.npy'},
     {'path': 'd7fa545d2b02aa8665af6e3f1d3fe6b5_10.npy'},
     {'path': 'cfcba9b44756c8f554fd595082c94f20_14.npy'},
     {'path': '13e8fe7ab882bea1f80c95762cc1a7da_17.npy'},
     {'path': 'aab7eef7ea7e2e9db115b76f7f27c1bc_4.npy'},
     {'path': '0c08a9202795762e5052ac8bfe9f9f16_11.npy'},
     {'path': '9618b92e6f3ce0d48550a555b1c00650_0.npy'},
     {'path': '0b316d22d962c21ab968ccd0d10ff41c_22.npy'},
     {'path': '86fdf2a92718e464e7abc55ed48c785f_5.npy'},
     {'path': 'b3d1610455e2860923475ebe2ee6c47f_53.npy'},
     {'path': '1425e54102c7d999613d0dcb09fab9c5_12.npy'},
     {'path': '75c3539e72676c5db1a3b4df08a514f5_15.npy'},
     {'path': 'e8cd20a79a3a3db029cdfb37cf0fc8d4_21.npy'},
     {'path': '4f5ce6facf6a9d0197ccbab1d1c5d95f_1.npy'},
     {'path': '77be54dcc43ccbe823db6459f3e02ea5_25.npy'},
     {'path': 'da3a0b0be4119e633ae934a2806e3431_16.npy'},
     {'path': '5eac829e16388a274112db0fa5874a5f_23.npy'},
     {'path': '3367ffd0c8ee6acfd7fb9c948eb7a605_18.npy'},
     {'path': 'd264d4920935c08823c78d7f37531685_4.npy'},
     {'path': '2eb66178e917e97d12cab7147613c1b0_9.npy'},
     {'path': '7f55a8197ac2de555bf49e6998cdb4b2_7.npy'},
     {'path': 'd4d9042a485f788eeb57678d17fff84b_11.npy'},
     {'path': '0a7e50bb3c8b9356022bd3189c873af0_43.npy'},
     {'path': '8c13297de8ac627dec0c0148f36bcd5a_3.npy'},
     {'path': '5ff9434d11bdc5406de6a7ccc747c556_53.npy'},
     {'path': '0a0a08cd3c5467e4787ef78051f0fa5a_7.npy'},
     {'path': '1659e25f849f1bcb60fc0756f7eec8f7_29.npy'},
     {'path': 'b593fbd8a5adca2c53edb69bae6aa720_12.npy'},
     {'path': '34163961c58d138f2359ae902ed2b20e_100.npy'},
     {'path': '385dd684f92d40dbbabd569d4246ce93_45.npy'},
     {'path': '9cba3077e4e154ac82baaeeb918fb3ed_10.npy'},
     {'path': 'f836f20cce743da4336ec86b37ca5397_8.npy'},
     {'path': 'd3e5b74ebe44dc32e7a02a0facfcfc69_40.npy'},
     {'path': 'd688ec555ac9d12def90ab70f0da1878_14.npy'},
     {'path': '345ec80eaa78761cd986c9acddf847a7_45.npy'},
     {'path': 'cf7bbf032a072228f27568ba52c383a6_6.npy'},
     {'path': '3c1c6d44b303fe8854f65b6a73811b9e_1.npy'},
     {'path': 'ff14dc6874974fd38755544a159f2055_9.npy'},
     {'path': 'ec0d1aaaac690d21a157981572cc56b6_7.npy'},
     {'path': 'c1637a3b9f3ee56431f111ea17a87fc8_13.npy'},
     {'path': 'e14c9a7a848a3383abc3b74f198f8415_11.npy'},
     {'path': 'be98963302532f69dc2427c134b74406_2.npy'},
     {'path': '7b6a3927dd01dcc24a2bfafbe085fe9a_83.npy'},
     {'path': '192f0c029b8b341a585932e0b4ff6105_35.npy'},
     {'path': '02f1a23c26d46789b67e15a111ab432f_12.npy'},
     {'path': '898af67b74fcd473444d4d7d79c5ae95_8.npy'},
     {'path': 'e65347011a70195d570771a9cde42683_13.npy'},
     {'path': '71482796464a0f3916b954af1cfc13d1_51.npy'},
     {'path': '0efd9a9003e6bff45b5347b8d9838463_21.npy'},
     {'path': '9cfde0249aba31663ad2f61b20410f09_8.npy'},
     {'path': '86f8907f68bc4a1a09a97fee4c138d65_29.npy'},
     {'path': '9512b0dbd760b0453ddf924138a7d184_41.npy'},
     {'path': 'dd6e83f248e0ada3f418aca847bfbd3e_4.npy'},
     {'path': 'f1c817510a0c87c3fe43e32a4f9146f4_61.npy'},
     {'path': 'fa43d8d460f7bde921e2cce850b2ccf7_33.npy'},
     {'path': 'ebf6d8bb0f2ce9b630efccfb582b3da0_1.npy'},
     {'path': '3510987c4e32c748fc02571bdb6dff46_20.npy'},
     {'path': 'c756c1eb429d5b17e63621559d5190f8_2.npy'},
     {'path': '59ddeab6ee858f0fcfbef366545d62ae_4.npy'},
     {'path': '46bbc4d484424b6de05a2270d11b7611_29.npy'},
     {'path': '718b0d71abf106b314cbe9a43c825df2_3.npy'},
     {'path': 'dd5812547a61cfe9dabfa486cf42f75a_35.npy'},
     {'path': 'e9a9ab5f255f20835dd70821de97f0e1_2.npy'},
     {'path': '3974276d1673fccd5a570fe347b87d46_61.npy'},
     {'path': '0872191523b3300cff0e4b46c6613abc_32.npy'},
     {'path': 'cd2086f56c056d131afb727cc2a28ad1_25.npy'},
     {'path': '2413898e5d40a7668b8c3d2268a96696_4.npy'},
     {'path': 'fe21680768342334914807dbba2227e4_15.npy'},
     {'path': '40a1154eaedd2c6c24cbab28dd1a4a4d_16.npy'},
     {'path': '89049329947e4ac3c32deb0db56055d2_7.npy'},
     {'path': '7cfb61eeafcdfbded50838daffc40b05_26.npy'},
     {'path': 'f55e99093e7afafc7637a694c47220b8_6.npy'},
     {'path': '12adf45b5c04827a1d1c777bd93b1c73_5.npy'},
     {'path': 'e24f68f0021bfb81b200c705e36d0e00_149.npy'},
     {'path': '76345540478ac4a56288f21011bbb5bf_6.npy'},
     {'path': 'c6c8d2384ec62d177c50a414a7758e75_3.npy'},
     {'path': 'f9d333b11f5328202b4d01b832c594e6_14.npy'},
     {'path': '23c567335b597ff1ba88e09240b9351f_16.npy'},
     {'path': '96e0eea72a6a6596edde5bfde4096d96_2.npy'},
     {'path': 'edff502f70d4b74f142e293a58c078ad_8.npy'},
     {'path': '43404cd5d904caefb5c9a079d0f44f40_9.npy'},
     {'path': 'e6cb4a9dd0ee5c914fcf281796839cf6_9.npy'},
     {'path': '2da71637c2b33701b620e53195f97958_78.npy'},
     {'path': '3c03c5feba370ee9af64f031fef0870e_21.npy'},
     {'path': '4134f8f18597df1d09c3d06184462d83_7.npy'},
     {'path': 'bc02c2354275f9dc8b743eaa1f6a4c99_20.npy'},
     {'path': '6304bdab6f475a1a9bec8af92254510e_33.npy'},
     {'path': '81c1dc08d1eed7fac126b6c196ec56d3_10.npy'},
     {'path': '735642ecdbd5cd62b69958470a8abc72_3.npy'},
     {'path': 'b2ef10111952092794e242f68ea76854_8.npy'},
     {'path': '6828a18487aa0a6bc9493752c57a19db_5.npy'},
     {'path': '99adf96d813aea080b10bfa31939e3ae_7.npy'},
     {'path': '8561881472bc89651af9f4a2dbb5dd30_6.npy'},
     {'path': '1c1f257e0f3b32749b6a58db107f5ad4_1.npy'},
     {'path': 'd52b6e78e4671857d9af351690d9a674_8.npy'},
     {'path': '5cb48daf9a199cfd7a30e9ccfc444c95_16.npy'},
     {'path': 'fa5c43bd945ffdb6e8efe8f255f37cca_1.npy'},
     {'path': '4f40b0bcd045d6cedc742bff953fa60a_6.npy'},
     {'path': '92a7c975549604ba4b6e3879c9a0c684_20.npy'},
     {'path': '274242660e325d25b9e31c298d3d91a0_5.npy'},
     {'path': 'c62c238ac64e7ba09989ded86c3d68e9_18.npy'},
     {'path': '6aacc729e778af6c882bddd24f6a48dd_10.npy'},
     {'path': '481912db4cece3024386a0e0617567d4_6.npy'},
     {'path': '22af56df776ca6d7a6582fbacb134275_38.npy'},
     {'path': 'f8466d3f7fa13683aa640cfe6bee8ebd_12.npy'},
     {'path': '4e57ac8d0f5e6cf07272602aaf442533_2.npy'},
     {'path': 'fe976be18266eefcc3a1a41950d4eb03_33.npy'},
     {'path': '4c4b4fafd55011214d8e96556afb5215_4.npy'},
     {'path': 'bd937cd7cd3f107c0927f282c0beaf58_14.npy'},
     {'path': '2bc4e496628637e10567dec6707eef31_6.npy'},
     {'path': 'a22ae497f7eafec7f81b4af3ed962f08_7.npy'},
     {'path': 'd5a5526b912dfb1b91e2b7cabf3f9e63_45.npy'},
     {'path': '1fdeb2ebc5126ce1406229a62e00ea80_24.npy'},
     {'path': 'cfc00ddc2f9d61c22e14bc23543a9a98_17.npy'},
     {'path': 'c76326ba4b69db4a1a3ae1b8ef704cff_102.npy'},
     {'path': '96c4e3c965773bd767de1ae7dfaed8f8_1.npy'},
     {'path': 'e9fd1d5f98ee2445532036fe144b420f_10.npy'},
     {'path': 'fb3550db295b4618da3fec76517251b2_11.npy'},
     {'path': '0d56976ad49fc68f28ea86a765f95725_54.npy'},
     {'path': '1a5833e4c0534c0132686c54bda1164d_16.npy'},
     {'path': 'b897c90c063ac8649f552a9d28560b3f_29.npy'},
     {'path': '8877107b01da3d76ed2f4c74487435f2_10.npy'},
     {'path': 'da4a8dd25db1bde2baf0b1ec4befbef7_110.npy'},
     {'path': '41bf2f510545dc0f5441798a493645fa_2.npy'},
     {'path': '37dcb28ecd18e9f02c7a3d2e34ea611b_4.npy'},
     {'path': '1787500bd2d43135a542b051fc394d89_3.npy'},
     {'path': 'd15e232794dff911e8f420194a5d7d9a_23.npy'},
     {'path': 'ba138cbf52478855f040267909d0d8a9_2.npy'},
     {'path': '2d36a1592bec5c1e0a51667703e3cb29_9.npy'},
     {'path': '3597d0b7b02ca702612809b095bf28c0_6.npy'},
     {'path': 'a64c9639ddd3e194a26759e04274d757_27.npy'},
     {'path': '3ae2ca18104a0033f17ff7831b10e361_56.npy'},
     {'path': 'c55220858c85888e0035f4aa0dcec04b_6.npy'},
     {'path': 'f1f641c87587d40dfb1e00e8fcbb7888_39.npy'},
     {'path': '5fb2688460ca0cd7054d2a8e133b03ad_12.npy'},
     {'path': '6fe9bfed8c168ff15a732587f7d945b5_42.npy'},
     {'path': 'c9a9a3832b049cf6df682046242ca6a9_1.npy'},
     {'path': '9a67126f3bea1318a787791623a7f434_40.npy'},
     {'path': '6988191b7fcaa8fb98c7a7876c299181_11.npy'},
     {'path': 'e155eb2b361478680c50a560d9e316a2_1.npy'},
     {'path': '54a3000bb382dd7eda0e3f4c88b9abe6_4.npy'},
     {'path': '2bcd9e8ec000c1185b2f12ac8f44260e_7.npy'},
     {'path': '48302376010ddf624548856b65c6d66c_6.npy'},
     {'path': 'fcae2f5788399ee8b4fcec17e10e2edf_21.npy'},
     {'path': 'c748b5ad7f6ea20a2ca77d24d6687bdb_19.npy'},
     {'path': '2736d1e5812c824776ca9d0c8fc2654c_29.npy'},
     {'path': 'b75365ba10a1109b61d2db0d9dd03647_19.npy'},
     {'path': 'bf2d55d612732a86706c0aeaea1670fa_31.npy'},
     {'path': '0911447492d8e47eb6a1fe0c413d5806_14.npy'},
     {'path': '3f1f86ec09149f259299a3f026368c75_88.npy'},
     {'path': 'd2a221f103fc24a70d0976139539a28b_60.npy'},
     {'path': '66765e4a1931cda6de63d71900b64fd4_19.npy'},
     {'path': '2fef96460f58e48e42f212ebdb29ced0_10.npy'},
     {'path': '630dcfb77175cbf319a8fa0d821e07ea_3.npy'},
     {'path': '79eb8258077c53bbd6f6c1b2eaba1549_4.npy'},
     {'path': '5c3aa8b942eb718ba7f7d1a97d288ae5_21.npy'},
     {'path': 'fd5d269e2b6f720dd6c183b0c9d8bf46_88.npy'},
     {'path': '822f5aa6ad301accfbab0b4139718a0e_31.npy'},
     {'path': '455edd254cdf786dd0e49bd2e00b8dc6_61.npy'},
     {'path': '2e9cd866acf6a52c783b692ae674f693_5.npy'},
     {'path': '8511e415230df72e80cbeebef146fe72_10.npy'},
     {'path': '79329e8dffb3f32631d1b2e3a05c0203_13.npy'},
     {'path': '21b064823f895b7cb9005c42a6f4d76d_15.npy'},
     {'path': 'c58f2fb01155c8d3856710bd5c4bc289_9.npy'},
     {'path': '83b70f1600800acfbbad259887da846c_7.npy'},
     {'path': '714f0581cc824d2607ec219a18a6063e_30.npy'},
     {'path': 'a782b186c48d9a509fe8cad0ef27850b_13.npy'},
     {'path': 'c379768082755252f2f6bc159ecaf981_2.npy'},
     {'path': 'dd0ffa5f2aaa7c4e050d45a9c5dd322f_4.npy'},
     {'path': 'da60ea3f671a2fca586b6a5c3c794898_13.npy'},
     {'path': 'bc2d88ef3c8b83b27931e24bfa495827_27.npy'},
     {'path': 'c0f0a4ead12733f446ce9cbe0cb8d7df_7.npy'},
     {'path': '4759433aa5cd534f408e9f6b8d581b28_26.npy'},
     {'path': 'ac66495b8ca78bceb0da0984a8798c08_2.npy'},
     {'path': 'da17e641ea9cfa7b1955ffd98dca8069_67.npy'},
     {'path': 'ba7fc90f3260fe0acc96dea1686ff385_16.npy'},
     {'path': '0a5a830c5c5fcdd4f773f0285a5eb551_11.npy'},
     {'path': '0d7cac8e0aa1bdb50ab14b67cee3b7b3_1.npy'},
     {'path': '717d49830d385ac40468cf7ccf14ea2c_11.npy'},
     {'path': '1dceeaa92746a19ed5ceb084b6b30196_5.npy'},
     {'path': 'fa13723aee6f617c290537c1563d8444_9.npy'},
     {'path': '4eff5367684372032fbc2ceb659a170b_31.npy'},
     {'path': '228c8ce32841aed42884048f815f6017_17.npy'},
     {'path': '7a8ddb01004a10c0d02d15f64821eb07_526.npy'},
     {'path': '70770ff1e43ff441e8da77281abd9e99_45.npy'},
     {'path': '9bf618ed040ead0336bd7bbd3508f239_5.npy'},
     {'path': '987696b1595ecf02ba5c1d31b1db6921_209.npy'},
     {'path': 'c3cebf0f885d7706cf36a23c5fee143a_29.npy'},
     {'path': 'a1ebf0edd2ec6706c47a91765d541396_12.npy'},
     {'path': 'f0955148390ac2b5d243ed60a72b2d02_38.npy'},
     {'path': '2ed23da352069b061e627d85f55f80b7_1.npy'},
     {'path': '6639385b30d30cab13c8a514ec882341_9.npy'},
     {'path': '15eedd97fb988e3a1ef2ba1b840125e8_11.npy'},
     {'path': '30c357a6b434bebc0eea5fae7c2f980e_59.npy'},
     {'path': 'f593db21fe68c341854f5e7487cc67e0_16.npy'},
     {'path': '894eae15c9580aab5ad3e0ca1a95c4dc_8.npy'},
     {'path': '5c88cf5725ff00b187ce23c2f836f42b_3.npy'},
     {'path': 'a2a3ef05b83c596c01ec39369743d8ba_14.npy'},
     {'path': '84a46bd14bef7e9b8f3cc6a210609084_3.npy'},
     {'path': '9a8b6dc050d33ea633952b2238cd1e4d_45.npy'},
     {'path': '5a6ab8f379b570ded5acad51f37aebe4_33.npy'},
     {'path': 'a2089037dbed567759b1b9e93f7037c9_8.npy'},
     {'path': '8255bf0d851e2fc05b6f1e3a4e45e336_2.npy'},
     {'path': '19e204cfb7d6ad20cba3346c15611678_100.npy'},
     {'path': 'd7317a771e1838ce77c67f61c091a9b7_9.npy'},
     {'path': '6b185eb2340cde8082a8af44b007d865_12.npy'},
     {'path': '899e1534cc9a86fe3dbfa95d522e3be6_13.npy'},
     {'path': '6327fd38f80beaded115e64c81f1c025_22.npy'},
     {'path': 'a84f24331d463250f4896b9ac5caac6d_15.npy'},
     {'path': '71b5facc64f8c2dd47e4eb2300f11bc1_58.npy'},
     {'path': 'beba872bf5755717d8102658627fa2aa_28.npy'},
     {'path': '85a1bf24e04601acfe1f503f8ac0d4f2_76.npy'},
     {'path': '81e4e50c94e8c61b6759ee8609f364e1_75.npy'},
     {'path': '4d81315df13b582edb4d07ee0518f2ed_16.npy'},
     {'path': 'ed3f30bf9ac5433ba78677a3d07428c1_11.npy'},
     {'path': 'ec3a93bd9bd2afa5ce40ebc879c8d78a_32.npy'},
     {'path': 'd32de9959a54309793e2dcee4513ff40_10.npy'},
     {'path': '2108c44d3d2588acf3909c89732d4437_3.npy'},
     {'path': '080e3376537e0f8928d6032821365a94_3.npy'},
     {'path': '4a2241755f374a7b5650fd363cfbeee4_21.npy'},
     {'path': 'a250c279546c62cd3ebf21753f6c267c_74.npy'},
     {'path': '6071d9fd108e89605e2ed98207887a42_123.npy'},
     {'path': 'ce42c3255c41e9694bed700d43eeb590_3.npy'},
     {'path': '71a36327a4e2309762407c8ac2f8b1f4_14.npy'},
     {'path': 'a6d1d23b756c815c9ff159b9783b0019_13.npy'},
     {'path': 'c89e5b9b8e40e4010da14cfc298676d4_10.npy'},
     {'path': 'fe06198067a653a6d82bebc8743769fe_37.npy'},
     {'path': '8e0965a3b6aa9f6b3d4f028cd75cb06a_152.npy'},
     {'path': 'b23ab208ccaca1325cc3a2b4a0f8e809_121.npy'},
     {'path': 'a399d5a26b5a05562bfbefb9520256eb_16.npy'},
     {'path': 'd2fd836f5eaca10bac9d2b14b05a9e5a_88.npy'},
     {'path': '21fc0c13805cf2e6110f358c65a32af3_86.npy'},
     {'path': 'e4c8df7f3afe75cb7129bb1a1b68f7db_11.npy'},
     {'path': '2afc2e992ff383fbeb624846816e713c_3.npy'},
     {'path': '492c30e5be2c7166df715762dad5afb6_5.npy'},
     {'path': '9a12b4de9a8ce66a64dbb80f7e38fa54_32.npy'},
     {'path': 'ee617ff22c5e422bc594299e0266efd5_33.npy'},
     {'path': '0cb9667161235e9bd30a44752c04c20a_31.npy'},
     {'path': 'fb41492854baafcf770af0ca826476e8_1.npy'},
     {'path': '8b7e9c04a094c8e5f6327a45912c4686_11.npy'},
     {'path': 'b7262412832843e93238d237044c7665_49.npy'},
     {'path': '5bd9d6c09a0c54c435a9e36e1f172b0e_66.npy'},
     {'path': 'd360800fb7f75a9ce7c984fce23953f6_16.npy'},
     {'path': '5578237f9d7caf26a2f3dd5f325b6d06_26.npy'},
     {'path': 'd54c187937d5a7cf46c692f76b2c29f9_5.npy'},
     {'path': '47ef9aceeed682e59b2e0a716362f175_74.npy'},
     {'path': '8dc910f7701ae64233de1d25b5a10693_31.npy'},
     {'path': '6e01bbee4c47c7dd416ff9fb288dd623_9.npy'},
     {'path': 'd67d741897c33736a60adcf4186cc150_63.npy'},
     {'path': '8e6808d8456e8ee2553a90a9bcf3f652_8.npy'},
     {'path': 'b053f349bc0320e84e4053c691399f80_7.npy'},
     {'path': '84c0c5fa9976ee81d40d9d296e1bc588_20.npy'},
     {'path': '64f83f6d95303d2bb449ea8d9e5301cb_110.npy'},
     {'path': '517c71029060575424212a6509343656_3.npy'},
     {'path': '104410266dd2cf0657dd6ae38dc8f148_4.npy'},
     {'path': '0f6c72c80c21a1dfd481bc056d4d9263_7.npy'},
     {'path': '78de89464b22e5888340f198a874bf9f_19.npy'},
     {'path': 'e08607455b9001e3c4b324197dabc6a2_5.npy'},
     {'path': 'd5e0dd9fba16f076391ca7d9fece596e_1.npy'},
     {'path': '3ceaa1508d7288a31a372f6fb152f331_4.npy'},
     {'path': '9184610c5b867df94ec9eefd7425232c_16.npy'},
     {'path': '5fa268416b369d86c2ef37972e23ef03_16.npy'},
     {'path': 'f408fd7ad601b38297d152e1a7918976_27.npy'},
     {'path': 'f5e6287fbf5eaf49f90604380e47fba5_5.npy'},
     {'path': '6d0a03534d1e831f97f2d2dcc7dfeec0_95.npy'},
     {'path': '0d161413156070432ea2414cf6fb7996_18.npy'},
     {'path': '6b4f11d768c1b3cdc3cdee1718ba10bf_5.npy'},
     {'path': '4d2fa1053d27e956a9d65886302b2c59_2.npy'},
     {'path': 'aef95ac204151c4df1d5e72f5ca31810_43.npy'},
     {'path': 'ad7bd8111ec39f812499832533401dea_9.npy'},
     {'path': 'bd4b2a0665aa21fd03f3a70452a11913_10.npy'},
     {'path': 'a03e8cd77ffc429c906b29156c423456_27.npy'},
     {'path': '9a7a45b4fb53dc365ff9aafd32c67987_312.npy'},
     {'path': '5b44b2db6d7509ad3552e6d9b7f3f397_8.npy'},
     {'path': '1764a9d0ee2d4a1a9d1e268657143778_21.npy'},
     {'path': '2d4f91c33052c0bfc70b85c1f9fdc040_101.npy'},
     {'path': '0c5c229ff427a498f8b73d8bc70fa4f3_6.npy'},
     {'path': '90a1f1dc8d238e8b4e6d89f7697169d8_32.npy'},
     {'path': 'f72c21baeca45995fa057adef2e27de4_9.npy'},
     {'path': '5ab35327dfe4d640d7986a926ba4ccd1_20.npy'},
     {'path': '4e9bbb222395e04de0992606bfc064a4_18.npy'},
     {'path': 'f114ad88911ec52aac0c5afa334a6bf2_22.npy'},
     {'path': '9cfc368d7870141781f08ef1ef707db8_12.npy'},
     {'path': '1fd7fe21bda1641284b9f2a4a0dc8b9c_13.npy'},
     {'path': 'bd4487b0e2e0e178e81a810221caa7ec_9.npy'},
     {'path': '48eb908dcc65b6722a150a08eb729f28_1.npy'},
     {'path': '378c2d8d74ff9152422e5279c05a2e66_2.npy'},
     {'path': '9a90b06be436f1ee21241e55e637248c_89.npy'},
     {'path': '20faf02403fbc471e49849725104ec51_12.npy'},
     {'path': 'e12d8d0310ba53d3fcc3791a449bc239_7.npy'},
     {'path': '812d376f97d8206f4baee033d29bb347_2.npy'},
     {'path': 'aab5dbbc0bec85eabd48615d3a919b4b_67.npy'},
     {'path': '18621cd3c12da2000ae83bbd3cfb3032_26.npy'},
     {'path': '1ea17ab6ca2348db080963935b5e0776_13.npy'},
     {'path': '171bbf1462f8d96bb8c968ae8767f728_5.npy'},
     {'path': '4361a85570eb2681bf130496e7f30175_4.npy'},
     {'path': '142977ece63f8bb49658934c5bec4ac1_4.npy'},
     {'path': 'f2c2bbc5616847fb2a88b05ed4005159_3.npy'},
     {'path': '553e6c98d615e5a6e34d006936f95512_14.npy'},
     {'path': 'd62fb8670737bec722a4a7373e4b2338_2.npy'},
     {'path': 'd14de0d25249e7333a9f26d3b270c4b1_6.npy'},
     {'path': '75c3539e72676c5db1a3b4df08a514f5_9.npy'},
     {'path': 'f6c7666cdfa484ace40a568f75f3ca73_8.npy'},
     {'path': '11e2b0f68638bf0892c3f64cf14b0b5d_20.npy'},
     {'path': '89be26bc22f094b2c6fb7543a13731c4_8.npy'},
     {'path': '38a6d395cee3c19e07af1c4c5ba6d5c6_11.npy'},
     {'path': '05d2abacf8a43c39067d8e73c60a9d80_6.npy'},
     {'path': '9ff08b125cee0fd6039905d2fb10526d_274.npy'},
     {'path': 'cdf798c53ecde5edcbb3d8f02528617c_7.npy'},
     {'path': '245abd2f351f356e8f1cd496d19f1075_53.npy'},
     {'path': '2b8a31a072386952277a2aa359728c24_13.npy'},
     {'path': '65df8c414948aa0e3ac282523892e5c6_4.npy'},
     {'path': '5b8af8ac2fe0caca86f303b43f671315_19.npy'},
     {'path': 'e6daa5a24c6e7e22cc8767447c235113_15.npy'},
     {'path': '37c67bffb99f2eb20b8d112827c51201_9.npy'},
     {'path': '21b7e84e227e75681cd406824a655bc7_9.npy'},
     {'path': '78c3c5e73043eb039b0e0eeb8305d6be_14.npy'},
     {'path': 'd5b808a009f806092ce4cafce86a38e9_8.npy'},
     {'path': '03b6128a29c7a87a6a3b08f899521a9d_12.npy'},
     {'path': '810887ee7372e9192c791f29dbda59f1_5.npy'},
     {'path': '5801d974a84c6231d1a05768392085cd_9.npy'},
     {'path': '5ea638915d1b992efe22d5a3dee25bef_31.npy'},
     {'path': 'a226a2bd858e8e5d278322c488c01edc_32.npy'},
     {'path': '10e1cf534e8896cd3075e92feb67dcf0_43.npy'},
     {'path': '977e876b4a5371f74475bfbb2564cad1_16.npy'},
     {'path': '0ea909525ad4c367e09f642da708a70e_9.npy'},
     {'path': 'b41f4b63a51ca1ef32ed584608787ad9_5.npy'},
     {'path': 'c3de6995600f09c4d3f4781ed7eb08fe_14.npy'},
     {'path': '52285d798a78e7c456ef6ef1c4c50192_7.npy'},
     {'path': 'eead8f64a3dbda87bd6d2c3b12b86e4a_309.npy'},
     {'path': '3f51012142c6d9eba675eeee198be4a7_37.npy'},
     {'path': 'fc2da419398bac011ee4bcbc51e0dabf_2.npy'},
     {'path': '66730ff73e2544ca3fec6e19f458fa10_37.npy'},
     {'path': 'a5ca5966b6265d8099ced1d1a4f1a7a1_12.npy'},
     {'path': '08367a25dfd59e374137bfd20590f2b0_9.npy'},
     {'path': 'b5320abba3a833f5472cc8a50e095aaa_48.npy'},
     {'path': '936c1965ffb29f01a46c6dbdc5d167ea_31.npy'},
     {'path': '4443aeb6badaf528eb522ad7ca9e3f06_10.npy'},
     {'path': 'b9c2b02181db14be3859c15419bc82ea_10.npy'},
     {'path': 'cebede0d100b50eb80f8a7dc2610bd3c_14.npy'},
     {'path': 'ccc39ec5062ea9c9b3e2749e68549728_10.npy'},
     {'path': '4813434d688c740765e75ac6fde8d378_12.npy'},
     {'path': 'b57677c29a50f977728665ed1c2f67df_2.npy'},
     {'path': 'b15d1f541badf1defa11d3fab0057898_9.npy'},
     {'path': 'a9525c62212d02d26a06ca9831fc0de2_41.npy'},
     {'path': 'a7659ab28750d5480d257669f0349ed7_225.npy'},
     {'path': '71492f58595f7c7f9e5c1e11920745ed_8.npy'},
     {'path': 'a3abd60607d466d57bdaadba7629deb8_25.npy'},
     {'path': '324a487ead6977b02c746d5117e9825a_16.npy'},
     {'path': '1d84f8387a71b601d7787417656d97df_7.npy'},
     {'path': 'a3f13b1d88432f87946cd8e818150385_8.npy'},
     {'path': 'b9bea7afd693b1bda4d9cdad366aaff9_11.npy'},
     {'path': '0ebe843b70e0cc5a5beaf59f73702dcf_23.npy'},
     {'path': '52aa32a90e17b3ab93a6134c5db22bef_24.npy'},
     {'path': '000e83c564317de4668c2cb372f89b91_6.npy'},
     {'path': 'e4c6a4bb8cbb2e329c1d410e25b9211d_86.npy'},
     {'path': 'a3e311ce1ef1ad3a02aec013edf21c86_32.npy'},
     {'path': '7b02a14a55f9c0a2dd30d1e1c892430a_8.npy'},
     {'path': '1743231b0e02a9d2da5410031bcffb42_36.npy'},
     {'path': 'e7d49a9f7a3c87d52895fb62e6dc6908_2.npy'},
     {'path': '6406cea0b01afff409c37d89b1c58a25_3.npy'},
     {'path': 'c0a69833edbd76748831ac2624051a69_60.npy'},
     {'path': 'f7759d071e51f6a9f1b101e62f5bcec0_18.npy'},
     {'path': '22bc3fb6e02386a1f3dd6337c25235d3_5.npy'},
     {'path': '0570c4c19c37da84439ee6fd069fc1b6_3.npy'},
     {'path': '4080b02e4cd90dc88f40483072fa2e73_38.npy'},
     {'path': '0b15a52e4686790ab631088865a206c8_2.npy'},
     {'path': '4fc46525a24188077558175a73948d1f_18.npy'},
     {'path': '617f80b64526f6e0adbe6a86cbae07fe_19.npy'},
     {'path': '7f026309be5c0e40ad91d123b7d50d8e_6.npy'},
     {'path': 'ea57c4e324dff8ff95cd329b7cc443c4_6.npy'},
     {'path': '56caed4d30a035c1c88157547f0a2bf7_36.npy'},
     {'path': 'd532d89168bfccf9de7de7585a6fd1b9_6.npy'},
     {'path': '370fdcf8858fbdfa20d0d38c6be96bf4_2.npy'},
     {'path': 'af944efc993b11ff0420aa1d4f7d2e35_3.npy'},
     {'path': 'a639a3888ce721ba0584c92e1731d487_30.npy'},
     {'path': 'fc8ba9e80e9d85cd7b9f8617a1ec59d7_55.npy'},
     {'path': '34853800b5e5b37a3f83ee6c684c0773_19.npy'},
     {'path': 'bf6565712602f41f27fe8e1cb6cb8dae_52.npy'},
     {'path': '7b81592c450bc6d1e7554e2346b5fe74_17.npy'},
     {'path': 'b79dfd1ef8327d6a293371c840832091_0.npy'},
     {'path': '1d214f0e2fe25d9ba70e33b17e60e0dc_2.npy'},
     {'path': '9e19232ac904169371ec36a919b75866_5.npy'},
     {'path': 'bd12a09aa889b14be7ee57200ab8305f_1.npy'},
     {'path': 'f0291e0fe6536f398a7a4a80c3e9ac22_2.npy'},
     {'path': 'cd7e7fb42cbc0a942f9b47e3ec94ed26_33.npy'},
     {'path': 'd08862c58eb441426f69efa2a67ac909_8.npy'},
     {'path': '176c2db55dcbd4f4c6157a0f18fc9284_12.npy'},
     {'path': 'ab55032f923589c8f47146b12425978c_13.npy'},
     {'path': '0624c1aeebf33c89a42a7f9597d47394_11.npy'},
     {'path': '63ffe64fa94913824ee7bb01dfb427b2_24.npy'},
     {'path': '26a8c9ffa78d693b60ba85c36bceb1e3_16.npy'},
     {'path': '49e526fddf026530cba3a1a6c0add3ad_4.npy'},
     {'path': '0a61a69efe77e0f4d8d9abcd26867a82_9.npy'},
     {'path': 'e14a1b7292e8751e705f92838ab6c121_18.npy'},
     {'path': '82d956b4c9546dba3bf6433d5081b58c_41.npy'},
     {'path': '1b842df0742e821fe380e4f53ee12815_8.npy'},
     {'path': '342d679011f218e796c09de56f7151dd_22.npy'},
     {'path': '5abb1f20f4eff121b6f097883a57cb84_17.npy'},
     {'path': '34163961c58d138f2359ae902ed2b20e_99.npy'},
     {'path': 'f421e2c5165625e11e18dfc239623a1e_3.npy'},
     {'path': '596c419ad687eebe5085ef47c42a4734_28.npy'},
     {'path': '09d1d57bf2fad842576d8f2f6d371b91_29.npy'},
     {'path': '3f9f9292593da47f715d4ddb8ec043c4_1.npy'},
     {'path': 'de9c4132359f14e29c39264bc93762ff_16.npy'},
     {'path': '88481d48c122e4cb00000627578e1c6d_43.npy'},
     {'path': '516b34f956fba22ffcd3567876636408_112.npy'},
     {'path': '5b10f3fc896a21dbdccef2df34c8553f_25.npy'},
     {'path': '41d2f030485e742610207a94e915c42c_1.npy'},
     {'path': '094ee4937da4eee33c519c40538c493c_16.npy'},
     {'path': 'c875b36cf2813fb362b78486c32a0a2d_4.npy'},
     {'path': 'f7b3e167ad44e42b730350cd608fbc65_21.npy'},
     {'path': 'b81d41bc3c6b700d6733bd061bb022f1_29.npy'},
     {'path': '927d8a7338ddb79f789480616912b571_62.npy'},
     {'path': 'e6e2714bc84e52e2817fb75346d5fed8_10.npy'},
     {'path': '519af903f91d06fb35c4676bfc770450_185.npy'},
     {'path': '498df361921b364978c8c7e4512cbbc9_8.npy'},
     {'path': 'cc7443b54a1c35acfbafb0e9e5af7c7a_51.npy'},
     {'path': '9644fb71fd3f3b25da7fd74dc9468bfd_12.npy'},
     {'path': 'f01f923307b223166c247e622bcbf460_15.npy'},
     {'path': '2e131aab303247d4451246fcbde01ad1_16.npy'},
     {'path': '4ca88f24ba6ab572d3ad81aeaf56b7d4_0.npy'},
     {'path': '3e6d57002256c871f9eb4f502a26c682_9.npy'},
     {'path': '41a6215742fcaa737d4df19457e00322_21.npy'},
     {'path': 'a45f7fbf97f2842e62f970d793456d20_3.npy'},
     {'path': '97deae14a00ada7f7096ae3e9b9eed1c_10.npy'},
     {'path': '86295f39c0891ce8cf9ef4ccbb0a7d97_93.npy'},
     {'path': '07640c9c6ed6c15edad21301c750d542_34.npy'},
     {'path': '53559f1116f302fa3e20bdc49d2c6a5f_1.npy'},
     {'path': 'd619a9a296e79b715a9ae691cf04f2ad_21.npy'},
     {'path': 'd8ab6b73be36df6dbb6165eb8015ec7e_6.npy'},
     {'path': '2f27cf99b46066fb482fe263d83ed4e8_13.npy'},
     {'path': 'e143910b0080122a68a9d112c67cd83e_17.npy'},
     {'path': 'e938687c8146ac781cce38dec42f9d78_69.npy'},
     {'path': 'e9d21cc441ff05b96a377c19a1154348_9.npy'},
     {'path': '658051a2410d1954839a92f70efcf43b_35.npy'},
     {'path': '1d36cfc140e6d1c8390d7b4868dee12c_30.npy'},
     {'path': '097b42becca811b3928bb5fdfba7c18c_86.npy'},
     {'path': 'a8b3c8f645eee366f0fb02bd8d263af2_3.npy'},
     {'path': 'ea2ef92b88337283f7cbc257c4810d27_34.npy'},
     {'path': '68f3ea240b76902bf8f3190a79f3d6cf_2.npy'},
     {'path': '4a8b1bf178cbc7ce447b5e0e4f8e4bf9_49.npy'},
     {'path': '476d0def39a517979084814e5e9353ca_29.npy'},
     {'path': 'c93edc8624f42f2d61a4a0a442b8bc3e_66.npy'},
     {'path': '83aba48dbdf6e8f6e05f67c8d2244f5f_9.npy'},
     {'path': '534dcda020481c4d96ea58711c2aa26f_16.npy'},
     {'path': 'edd4024503694beda3c5a5f4fde5346e_8.npy'},
     {'path': '54ee50c3f8ddce5fbbb6dab8fe1cda57_8.npy'},
     {'path': '348ff6441178744d6dad6288f66dd63a_87.npy'},
     {'path': 'd73ce78feb80e4557b167861ae93543b_92.npy'},
     {'path': '78b61f374107e554d08ca8f901e8c531_2.npy'},
     {'path': 'f66983c2d9158f8dd2a228cf08978984_40.npy'},
     {'path': '794e320df42a20059c56f167a2176be4_93.npy'},
     {'path': '3318c96ca215094565ac69254b3537a7_37.npy'},
     {'path': 'db2c4fcaddbbda9dbccd8e3e81cc3325_3.npy'},
     {'path': '23fe6d3c6de10f6c985a7038632b46a1_12.npy'},
     {'path': '20da6709f62630887f46fefcc2196457_3.npy'},
     {'path': 'c7725ff0250c6125d51e18d8e5c1d816_2.npy'},
     {'path': 'd51b0c44cd6d666cf8e16984ea800bea_21.npy'},
     {'path': 'f944be58d0024496ea6c8707df32c97e_1.npy'},
     {'path': '9e86fa170bf349e7d90617a0964af334_39.npy'},
     {'path': '43874dcd1a6e0d2f91104ee8c494975b_8.npy'},
     {'path': '0f645f47c44b403f86ab08faee61cd49_3.npy'},
     {'path': 'a05bfb8c249d87515f2eeaa0911bd60e_22.npy'},
     {'path': '0ad3fc991e4fbd92cc7b46be1c072140_33.npy'},
     {'path': 'd40404b931af5e240749ea0382e09d57_3.npy'},
     {'path': 'b133973eef8d084f851826922cd0cbfd_4.npy'},
     {'path': 'a313413990009ee6eba6c58203ef38bd_3.npy'},
     {'path': 'b36b41cea72c79961a2804b57a40aa70_2.npy'},
     {'path': 'd6b8c370f880f04a69fb6ceb68066d30_8.npy'},
     {'path': 'bb28e4747604f3f4c5c9854145b89e6b_6.npy'},
     {'path': '506a25977049b554712ebca03e4f7922_8.npy'},
     {'path': 'd098ac58eb57f7269d3eeefe5828d723_53.npy'},
     {'path': 'b8920be6db33fcbeb37e973b3803923b_12.npy'},
     {'path': '12a1b3df9fed144a13600f13c0f23777_32.npy'},
     {'path': 'bde2f303040bbda39edd60e2b20dc1be_26.npy'},
     {'path': '4f2c0ea509187d14647542e27d6e4f3a_193.npy'},
     {'path': 'd4792795afb2223f73fb891da47fa28f_5.npy'},
     {'path': '7c1a53379eda034b4d16d067bf8993b5_26.npy'},
     {'path': 'd8446d4be243260a4f63d815ec97e619_1.npy'},
     {'path': 'c56d45634882de4be4a5b9e25796505a_13.npy'},
     {'path': '70488e9a2a9f3d10e639723c7828e172_3.npy'},
     {'path': '6deb7afb675adf12dffc03e4619cc663_9.npy'},
     {'path': 'dcf84a43279b5ed33f9c4d809b147985_30.npy'},
     {'path': '85a94c64b723f96bcc1f74943b261480_19.npy'},
     {'path': '0722a2c48f9f917e16e2dd4dd0682a52_12.npy'},
     {'path': 'b2364fdc809e37a30889ae21b04a53fe_53.npy'},
     {'path': 'e9ca26c27a5e5888ba96f3905369303d_5.npy'},
     {'path': '0ec6c3635d9f0c5cef53d2221a8a1ff2_62.npy'},
     {'path': '864868ac3cd716b621a4d72600ec808d_10.npy'},
     {'path': '6e974ef86f60e58a23fb4c016bc3da64_46.npy'},
     {'path': 'dca86b88aa11e8c5a142694803b7f510_52.npy'},
     {'path': '6ffdba35f00747b513cc8f86fc22d0d8_19.npy'},
     {'path': 'c2ed79fa7ceb2fdb348204bb1e47c227_5.npy'},
     {'path': 'faea0722f710e84b2b952e29e0de2954_7.npy'},
     {'path': '2ec7201b63297ddd0abd3db978f140c5_10.npy'},
     {'path': 'bd8b88d9abe19e6aab1aa8f87beccccc_3.npy'},
     {'path': '639884efc3d428efc64366bfc69bd82d_16.npy'},
     {'path': '48559f386c366a6391eecee4cd257aaf_10.npy'},
     {'path': '0a1f1a9da0320890c412fcb73102c02f_8.npy'},
     {'path': 'fa6fdea936f62a7e9816f29371458a16_1.npy'},
     {'path': 'f503d8739a332563b3376582a7f0c337_4.npy'},
     {'path': '64837288c46ee6ffed8c7c603c19d3d5_25.npy'},
     {'path': '45294d42ac7367d5407f76da90acf79b_38.npy'},
     {'path': 'c1686aab3da8098917c7e5c3f042ab09_7.npy'},
     {'path': '26a61361cab2c370ee8ba9d04f534108_27.npy'},
     {'path': 'ec3d9fa0f3aa586651f5f9a00a1f2f93_21.npy'},
     {'path': 'bc1edb8aaa39818b29c5a5c9e3735b4e_3.npy'},
     {'path': 'b793281b15fc186f5803a2739728a96a_39.npy'},
     {'path': '2efc3a866d6c34aa1aeaa1a577820bcf_249.npy'},
     {'path': 'c37da678dd2ca163c259413260068769_18.npy'},
     {'path': '8f55fe77d423c09a9fea129930c5aea7_20.npy'},
     {'path': 'd0f935ac5ead04a8790850f38831d48a_28.npy'},
     {'path': '52902c6e01515e328206708a38cfe958_5.npy'},
     {'path': 'e6fda348dbe961495fc65d911284a92a_12.npy'},
     {'path': '75d64de35e25ecab6c884ed471003d7a_114.npy'},
     {'path': '60b5574e9a1493648be7142606d5bafb_17.npy'},
     {'path': 'ab10e0c70341dd2a5538ad85f3059491_4.npy'},
     {'path': 'af5a3f3c5faddf5abf00ceab74da7053_4.npy'},
     {'path': '2488cfff599c11f438629ca0595128b3_63.npy'},
     {'path': 'c49cb56f7f51cf77c8e0f3217cb9abaa_0.npy'},
     {'path': 'de30cff9ce59a8234985daf906009925_10.npy'},
     {'path': 'a516e3ad070fde8e081b56676b05a3ed_25.npy'},
     {'path': 'e2db6ade1012319d04d2c318d4c100be_10.npy'},
     {'path': 'a1b5b38746fa923bad1ad6bf26a47a8e_30.npy'},
     {'path': '4401fe5ffb6b6d9fe9c492f83dfa4a7c_7.npy'},
     {'path': '1146dd3c369fa133ae673fbf6fbe6a9f_4.npy'},
     {'path': 'f1cf1b58dc10ef32261890f3296209ca_35.npy'},
     {'path': '347c52301ebab6999d5326920cdf94e7_26.npy'},
     {'path': '23aafe7980fd1582ada6cc45c5113461_4.npy'},
     {'path': '2ef07b4fa2ec545e7e01e8156c5ae858_7.npy'},
     {'path': '0c6b7ab1a9bec61a5f4fd29a05a1ab31_3.npy'},
     {'path': '6739904428f260b17807265ad4178053_26.npy'},
     {'path': 'e41b8da316e0018c1758901a646df924_4.npy'},
     {'path': '0686354e90af64cf081e9b217ffb64f0_84.npy'},
     {'path': 'c471bb610266d858d35bd73d591d255a_11.npy'},
     {'path': '6646d8b56c20a6a43252f5f8a2633b17_13.npy'},
     {'path': '7cb24616d64aefbda86eba7e8637af02_26.npy'},
     {'path': '5a6638592b3c9f5ff2f67cd38a72cdc8_19.npy'},
     {'path': '55074bf59c246a38e4c535d269032e19_2.npy'},
     {'path': 'dc0f4795808e22089b36f92ccee06e91_23.npy'},
     {'path': 'c7c7ea4f221a3ee8b01956c3b7623f11_52.npy'},
     {'path': 'b33bdcdfcad7f93b7d5e8d91517a2f9d_5.npy'},
     {'path': 'de0b1c71e9c67ef0c166ee5084f4bb19_23.npy'},
     {'path': '16a535aa8c82b232c3078de5b0724b54_29.npy'},
     {'path': '63a01d6b141b2952cb015b5e41d186fb_11.npy'},
     {'path': 'fb47cdca495b9f93fa0a02e82745c280_9.npy'},
     {'path': '3830d5c5645c712a312f861b1d88b69a_33.npy'},
     {'path': '3bfae85ff0fde2ec1ad581a03040a2f6_14.npy'},
     {'path': '02b3a9720fa36e3bcb489a4a3d3b3547_21.npy'},
     {'path': '4a0da50fde7b77858c9e93b75fabf735_10.npy'},
     {'path': '84e034c1cf1c6484cb847b0ae7c67aab_1.npy'},
     {'path': '9b2533d417017924c5e74db4d6fce5f6_16.npy'},
     {'path': '3debf911cea461bb74e1022e4120febd_23.npy'},
     {'path': 'c97e21b1b14f6a31e992abb533827132_8.npy'},
     {'path': '2a67337f0ca392035ee486f4619302ea_9.npy'},
     {'path': '59f9e8d81648f49ebc9c5f0084e9cd2d_4.npy'},
     {'path': '1db4ebd4257d92e472b0579840567f6c_21.npy'},
     {'path': '1d48d4bb153e8a154399e25a0adc1c81_18.npy'},
     {'path': '066a9c58bf9dd3675527f0071a73e419_11.npy'},
     {'path': '9377c8c3324f6fc45b0ba46860837fc5_13.npy'},
     {'path': '0eef7d0562d2a57ca44a90e01d6c08db_9.npy'},
     {'path': 'b6ac26544839b2f6272a52dcfcb1d05b_3.npy'},
     {'path': 'd2055b97ad4d20d0a4a66ffb36fa59d9_3.npy'},
     {'path': 'c07f653b6602fa5a5fdf05df3c7832f8_6.npy'},
     {'path': '02fddb539c0dc535f8cf60d1da3fa82f_31.npy'},
     {'path': 'a59040c68555757374a37bd2c75b82a6_35.npy'},
     {'path': 'de2fae07aae0e330e67b7ad82f535f3f_7.npy'},
     {'path': '0baa70401fdde86b3054c30797ce0ee6_15.npy'},
     {'path': '922cba18e3dae69909124352a98fdb5b_21.npy'},
     {'path': 'f3759648e0764ad1ab59d602c5720a6a_9.npy'},
     {'path': '8be899d586e55afc67de4e74a43e47e7_16.npy'},
     {'path': '7b40f23478f0bf3f0430c23e67c18cce_1.npy'},
     {'path': '3575d30f4480ce852aeb2565a02cb825_17.npy'},
     {'path': 'f9674391f5eb6ac57634590f8e637a07_1.npy'},
     {'path': '103ce12642ea295a7c4bc97387e0ca1a_9.npy'},
     {'path': 'e569bceb8bab73a09f746da53f09ae99_2.npy'},
     {'path': '944681a1ebf03086af29a33186b58f24_22.npy'},
     {'path': 'df0e0ac89ae5c4423e59d4f7f2e994e3_6.npy'},
     {'path': '921d48d9292c561cde41610662aa3d4f_1.npy'},
     {'path': '4266112e617d8cde7c90a38c526135c7_4.npy'},
     {'path': 'f81a49f5a34c6eb5cea18f8eeeaab2eb_9.npy'},
     {'path': '14a2404e673e87a035141b3ff3ef30ba_86.npy'},
     {'path': 'c3dd53b5c4e865d7cf0d5325e2a026c8_7.npy'},
     {'path': '938174500a1fc3d6041780a7b43bc3a1_31.npy'},
     {'path': '30fe39ec79a74f5e22b824147dafb0ae_17.npy'},
     {'path': 'b0596c7f7151abff868474925b6310c1_59.npy'},
     {'path': 'fba5d4c99011e9dea3763b4368a3a437_15.npy'},
     {'path': 'a641777f9180fc6bf3b3c91dd2a1e667_13.npy'},
     {'path': 'afc8c7c28802bf4d4de8ad7b136f7bd7_9.npy'},
     {'path': 'dedd63c4744a6cf327ad62b1e41f7d83_13.npy'},
     {'path': 'b86398b9e938bbdb4641eafe7b3b6e0e_10.npy'},
     {'path': 'fc48e99911dfd88394605700e443806d_12.npy'},
     {'path': '26714866d51bb2e56120da02ebc029fa_60.npy'},
     {'path': '51af37fbeb09fd28d2221828ca3ad949_61.npy'},
     {'path': '721565e69b51a03c3e849b4b9f9bc77b_66.npy'},
     {'path': 'a0ed9dd770ad841531c9b084c1e3ec07_2.npy'},
     {'path': 'acc34491983ade09daa50a210d367ff8_8.npy'},
     {'path': 'ba5ddd23cc7ad2e18daba2e4c7dfa7e7_17.npy'},
     {'path': 'aac1cf123e47aaa2303076b2db9a31c6_5.npy'},
     {'path': '7c5962e05747468dcb4325ef510bea29_29.npy'},
     {'path': 'd7e66707c793ade272f7106199ac6ef8_19.npy'},
     {'path': '3009a84027c18605fc5c92f9f9efd530_10.npy'},
     {'path': '31f07d6b09fdea89f21d8baf745c6c61_3.npy'},
     {'path': '5b014378cf5b71d069fd6b5119ff24d5_9.npy'},
     {'path': 'f6f23c7866c25f95abf3c0a5bbc9689d_7.npy'},
     {'path': '990e60a5aea5b053d31ca7d7b6712601_37.npy'},
     {'path': '06e8fcf12f0c95bc712fbb6cd5cb95c3_12.npy'},
     {'path': '190bd8cec69e7a1f8a14ef6bca785b50_8.npy'},
     {'path': '4a157c474f2c0c4f03235856917eac97_56.npy'},
     {'path': '29207863f54a32d15646ee1db3c372c7_12.npy'},
     {'path': 'df319024f13c2c8f04588f9170a8858c_23.npy'},
     {'path': 'f31214f1498f5bb6796125cc8c0bb9d1_14.npy'},
     {'path': '297a609fabe5979f7257307fe61e1145_14.npy'},
     {'path': '68dd01fbc2b6945a2a8a6886dab98e20_4.npy'},
     {'path': '1e7b8a37861c85a75d64c2cabd23b17f_61.npy'},
     {'path': '3dec0d5e4acb7f0a3f0c6b5ebd5d4d7d_13.npy'},
     {'path': '1360854fe2c1f3ff50834f643ac6e108_59.npy'},
     {'path': 'cdfe271c36d2d7740dd22010281f3af4_73.npy'},
     {'path': 'd55b5032ab7b0698cbfe303d58513590_3.npy'},
     {'path': '57c8b8ccc977baa86f705c00d087d832_214.npy'},
     {'path': 'c6a40b2a5b37395a9df933f05f59f91c_8.npy'},
     {'path': 'f4e617a6fb4622af9d2a7f8ffda62033_21.npy'},
     {'path': '8511e415230df72e80cbeebef146fe72_7.npy'},
     {'path': 'b55ea9a4c4b046b0d4e5f02930031ca6_14.npy'},
     {'path': '3d85cae98c529b1df14fe1750138f286_15.npy'},
     {'path': '01782d5c4bfd3a9ca7a22aa50d619cda_38.npy'},
     {'path': 'fa3c01b80ceb256d444f121ab4b28a4d_14.npy'},
     {'path': '8715276d405e7994a7449c552d17491e_7.npy'},
     {'path': '4079573dd78845c9f2d2b88aebc6e8b9_18.npy'},
     {'path': 'be0a17c2c922d974a4abb955080cc19b_87.npy'},
     {'path': 'b04cdcc4ed2f0da0e0cf2f556652fd94_10.npy'},
     {'path': '2ffe64a23400f45f85da0c08a28eacbd_19.npy'},
     {'path': '5e020f16dac3bdf82fe03ac4af14f0b2_33.npy'},
     {'path': 'a6c71278c7579ee3ae3449a506c9f21d_11.npy'},
     {'path': '327535a639b9165e32d3bcbed3fd9b48_23.npy'},
     {'path': 'd25a7855f351aaadcd04f5b4f8e2399f_7.npy'},
     {'path': 'bced94d8c9e14f85754bf7e0d511f3af_137.npy'},
     {'path': '7074c0afe95b48fda5b665c01b54730c_43.npy'},
     {'path': 'b763c7493726924f6fc2aa1621cd80f4_11.npy'},
     {'path': 'dbf32ebd7b7f7b78f95e2f0f99208d42_21.npy'},
     {'path': 'c53816f83ba2f6ef098143434a77c81e_18.npy'},
     {'path': '3e613deae71fe34a6c6bbd0ca4e6c693_30.npy'},
     {'path': 'cbace029d56058d22e92c25f9e2e757d_3.npy'},
     {'path': 'c72839cbd8db351055625a161960988f_21.npy'},
     {'path': '5ac217a6f9a2098d02511e4fb72dc6be_29.npy'},
     {'path': '6a15dc9e051350f0e8e7ea77a96f2a97_9.npy'},
     {'path': '80430431aed97bb5ff15ce13fe758c5a_42.npy'},
     {'path': '8d7d94825d7e223083a1261736c4c627_8.npy'},
     {'path': '540f374c1f6b644eedc9d7662f77b0d0_85.npy'},
     {'path': '9bec01bf7105a72a23fc12fc8417074b_37.npy'},
     {'path': '33846b633eca89871a98e83102f6e6bc_3.npy'},
     {'path': 'b55a138b82f2f6d723ece48f089e9072_4.npy'},
     {'path': '4e8c3a467f0ee116f83858a1a3915ac6_10.npy'},
     {'path': 'ac2bcaafb2cde8a41ad58f85b1c48154_4.npy'},
     {'path': 'c0f1ac347eb2fb8cb882a262093b699b_3.npy'},
     {'path': '6993f113849f65093d7a44bc4edbbee5_15.npy'},
     {'path': '2c868da807a46924ec312da8a1d73b9b_5.npy'},
     {'path': 'a26dc5139fb59cd4edfcc585f9cecebf_13.npy'},
     {'path': 'a1750ce85fd1520c0739e893eba4fbca_24.npy'},
     {'path': 'b80511f8b8c15c9529735f113b93faf2_5.npy'},
     {'path': '64f83f6d95303d2bb449ea8d9e5301cb_39.npy'},
     {'path': '3f97d0d19ff223ff6e70f3d0f9f8ab9a_6.npy'},
     {'path': '4b021d2cdc6eba9265c54ce3a4bb4dc3_1.npy'},
     {'path': 'b6404c6a7749e2bd47d0871ad28b355d_94.npy'},
     {'path': '5522745d8864f3c125fb63d2c38333df_13.npy'},
     {'path': '6977af416f26551e520d0ec0bbb0cb0a_2.npy'},
     {'path': '268b020d09ef8ea20784e00f6eb08684_28.npy'},
     {'path': '90590d19797bc57346d84fec4813a047_6.npy'},
     {'path': 'c9f680122d5beacab7d80f5f9493efbc_7.npy'},
     {'path': '497b137c7ff3bab0c86e1182e88d6055_7.npy'},
     {'path': '2d9bf1b5f64473a1beac6311f93ff634_21.npy'},
     {'path': '856ebefd3f9790fa34077fec3471683f_71.npy'},
     {'path': 'a404de8652832325fec1e924948311d2_42.npy'},
     {'path': '0a055c9978adbed81194a828242ec605_2.npy'},
     {'path': '4ac332ef9b3a61d1eda86790fe720c8e_5.npy'},
     {'path': '758d5cefceaa52f8743ec8abf568e65f_1.npy'},
     {'path': '2a4c9b586c50d3e05e87f3dc1c52a0e5_17.npy'},
     {'path': '90a087734b92995e2e194aa75973f88b_24.npy'},
     {'path': 'c130fa14ac0b16291f0b5d938485dc77_14.npy'},
     ...]




```python
np.random.randint(0, 2, (5, 5))
```




    array([[0, 1, 0, 1, 0],
           [1, 1, 1, 1, 1],
           [1, 1, 0, 1, 0],
           [1, 1, 0, 0, 0],
           [1, 1, 1, 0, 0]])




```python
plt.imshow(donut[25:, 25:])
```




    <matplotlib.image.AxesImage at 0x7b93d186ec80>




    
![png](video-compression-old_files/video-compression-old_37_1.png)
    



```python
a = np.arange(36).reshape((6, 6))
for i in range(dims := len(a.shape)):
  a = np.concatenate([np.expand_dims(np.take(a, indices=0, axis=i), i), np.diff(a, n=1, axis=i)], axis=i)
print(a)
for i in range(dims-1, -1, -1):
  a = a.cumsum(axis=i)
print(a)
```

    [[0 1 1 1 1 1]
     [6 0 0 0 0 0]
     [6 0 0 0 0 0]
     [6 0 0 0 0 0]
     [6 0 0 0 0 0]
     [6 0 0 0 0 0]]
    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]
     [18 19 20 21 22 23]
     [24 25 26 27 28 29]
     [30 31 32 33 34 35]]



```python
# np.array(np.array_split(np.array_split(a, 3, axis=0), 3, axis=2))
# np.array(np.array_split(np.array(np.array_split(a, 3, axis=0)), 3, axis=2)).reshape()
```


```python
[1, 2, 3, 4][slice(0, None)]
```




    [1, 2, 3, 4]




```python
# np.vectorize(lambda x: np.array([x]*4))(np.arange(25).reshape((5, 5)))
np.reshape(r := [np.arange(25).reshape((5, 5)) for i in range(25)], (25, 25))
print(r)
b = np.empty(25, dtype=object)
b[:] = r
# print(np.block(np.reshape(np.array(r, dtype=object), (5, 5))))
# np.block(list(map(list, r)))
print(np.block(b.reshape((5, 5)).tolist()))
```

    [array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]]), array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])]
    [[ 0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3
       4]
     [ 5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8
       9]
     [10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13
      14]
     [15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18
      19]
     [20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23
      24]
     [ 0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3
       4]
     [ 5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8
       9]
     [10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13
      14]
     [15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18
      19]
     [20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23
      24]
     [ 0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3
       4]
     [ 5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8
       9]
     [10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13
      14]
     [15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18
      19]
     [20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23
      24]
     [ 0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3
       4]
     [ 5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8
       9]
     [10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13
      14]
     [15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18
      19]
     [20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23
      24]
     [ 0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3  4  0  1  2  3
       4]
     [ 5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8  9  5  6  7  8
       9]
     [10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13 14 10 11 12 13
      14]
     [15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18 19 15 16 17 18
      19]
     [20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23 24 20 21 22 23
      24]]



```python
np.pad(np.arange(25).reshape((5, 5)), 2, mode='reflect')
```




    array([[12, 11, 10, 11, 12, 13, 14, 13, 12],
           [ 7,  6,  5,  6,  7,  8,  9,  8,  7],
           [ 2,  1,  0,  1,  2,  3,  4,  3,  2],
           [ 7,  6,  5,  6,  7,  8,  9,  8,  7],
           [12, 11, 10, 11, 12, 13, 14, 13, 12],
           [17, 16, 15, 16, 17, 18, 19, 18, 17],
           [22, 21, 20, 21, 22, 23, 24, 23, 22],
           [17, 16, 15, 16, 17, 18, 19, 18, 17],
           [12, 11, 10, 11, 12, 13, 14, 13, 12]])




```python
np.arange(50)[slice(1, 20, 3)]
```




    array([ 1,  4,  7, 10, 13, 16, 19])




```python
# np.indices((5, 6)).reshape((30, 2))
np.indices((5, 6)).transpose(1, 2, 0).reshape((30, 2))
np.indices((3, 3, 3)).transpose(1, 2, 3, 0).reshape((27, 3))
```




    array([[0, 0, 0],
           [0, 0, 1],
           [0, 0, 2],
           [0, 1, 0],
           [0, 1, 1],
           [0, 1, 2],
           [0, 2, 0],
           [0, 2, 1],
           [0, 2, 2],
           [1, 0, 0],
           [1, 0, 1],
           [1, 0, 2],
           [1, 1, 0],
           [1, 1, 1],
           [1, 1, 2],
           [1, 2, 0],
           [1, 2, 1],
           [1, 2, 2],
           [2, 0, 0],
           [2, 0, 1],
           [2, 0, 2],
           [2, 1, 0],
           [2, 1, 1],
           [2, 1, 2],
           [2, 2, 0],
           [2, 2, 1],
           [2, 2, 2]])




```python
# (1, 2, 3) % (2)
# type(np.array([1, 2]).shape)
# np.full((5, 5), 3).shape % 2
```


```python
plt.imshow(np.gradient(donut)[1])
```




    <matplotlib.image.AxesImage at 0x79ee17770100>




    
![png](video-compression-old_files/video-compression-old_46_1.png)
    



```python
a
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-3f786850e387> in <cell line: 1>()
    ----> 1 a
    

    NameError: name 'a' is not defined



```python
compress_bytes()
# automatic inverse function generation?
```


```python
print(pickle.dumps([1, 2, 3]))
print(pickle.dumps(np.array([])))
print(pickle.dumps([]))
print(zlib.compress(pickle.dumps([[i] for i in range(30)])))
print(zlib.compress(pickle.dumps(list(range(30)))))
print(pickle.dumps({5: 30, 6: 40}))
```

    b'\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00]\x94(K\x01K\x02K\x03e.'
    b'\x80\x04\x95\x88\x00\x00\x00\x00\x00\x00\x00\x8c\x15numpy.core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x00\x85\x94h\x03\x8c\x05dtype\x94\x93\x94\x8c\x02f8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x00\x94t\x94b.'
    b'\x80\x04]\x94.'
    b'x\x9c%\xc4\xa7\x15\x800\x00@\xc1\x00\xa1\xf7\xde=\x8aE\x98!"\x82=\x18"(\x96\x85\xff8q\x97\xbc\x1f\xf1SfS\xe6\x10\xfa\xcb"\x9b\x1c\x92\xe4\x92G>\x05\x14RD1%\x94RF9\x15TRE55\xd4RG=\r4\xd2D3-\xb4\xeas\x7f\x01\xd0\xe75z'
    b'x\x9c%\xc1;\x02B\x00\x00\x00P\x15\xa2P\xc9/\xb47\xb9\x83\xf9\x9d\xc1\xe8\x1e\x1d"\xf75\xf4\xde7\xdc\xe6\xe0o\xf9}\x04\x0e\x8eNB\x91\xd8Y"uq\x95\xc9\x15n\xee\x1eJO\x95Z\xa3\xd5y\xe9\rF\xefu\xda\x01\xf3j\r\x84'
    b'\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00}\x94(K\x05K\x1eK\x06K(u.'



```python
donut.shape
```




    (10, 10)




```python
[2, 1, 0] in np.random.randint(0, 3, (20, 3))
```




    True




```python
# recursive rules?
```


```python
np.unique(np.random.randint(0, 3, (50, 3)), axis=0, return_counts=True)
```




    (array([[0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [2, 0, 0],
            [2, 0, 1],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 0]]),
     array([2, 1, 3, 7, 2, 1, 5, 2, 1, 2, 1, 1, 1, 2, 3, 1, 2, 4, 3, 3, 1, 2]))




```python
dx, dy = donut.shape
xs, ys = np.mgrid[:dx, :dy]
np.column_stack((xs.ravel(), ys.ravel(), donut.T.ravel()))
```




    array([[0, 0, 0],
           [0, 1, 0],
           [0, 2, 0],
           [0, 3, 0],
           [0, 4, 0],
           [0, 5, 0],
           [0, 6, 0],
           [0, 7, 0],
           [0, 8, 0],
           [0, 9, 0],
           [1, 0, 0],
           [1, 1, 0],
           [1, 2, 0],
           [1, 3, 0],
           [1, 4, 1],
           [1, 5, 1],
           [1, 6, 1],
           [1, 7, 0],
           [1, 8, 0],
           [1, 9, 0],
           [2, 0, 0],
           [2, 1, 0],
           [2, 2, 0],
           [2, 3, 1],
           [2, 4, 1],
           [2, 5, 1],
           [2, 6, 1],
           [2, 7, 1],
           [2, 8, 0],
           [2, 9, 0],
           [3, 0, 0],
           [3, 1, 0],
           [3, 2, 0],
           [3, 3, 1],
           [3, 4, 1],
           [3, 5, 0],
           [3, 6, 1],
           [3, 7, 1],
           [3, 8, 0],
           [3, 9, 0],
           [4, 0, 0],
           [4, 1, 0],
           [4, 2, 0],
           [4, 3, 1],
           [4, 4, 0],
           [4, 5, 0],
           [4, 6, 0],
           [4, 7, 1],
           [4, 8, 0],
           [4, 9, 0],
           [5, 0, 0],
           [5, 1, 0],
           [5, 2, 0],
           [5, 3, 1],
           [5, 4, 0],
           [5, 5, 0],
           [5, 6, 0],
           [5, 7, 1],
           [5, 8, 0],
           [5, 9, 0],
           [6, 0, 0],
           [6, 1, 0],
           [6, 2, 0],
           [6, 3, 1],
           [6, 4, 0],
           [6, 5, 0],
           [6, 6, 0],
           [6, 7, 1],
           [6, 8, 0],
           [6, 9, 0],
           [7, 0, 0],
           [7, 1, 0],
           [7, 2, 0],
           [7, 3, 1],
           [7, 4, 1],
           [7, 5, 0],
           [7, 6, 1],
           [7, 7, 1],
           [7, 8, 0],
           [7, 9, 0],
           [8, 0, 0],
           [8, 1, 0],
           [8, 2, 0],
           [8, 3, 1],
           [8, 4, 1],
           [8, 5, 1],
           [8, 6, 1],
           [8, 7, 1],
           [8, 8, 0],
           [8, 9, 0],
           [9, 0, 0],
           [9, 1, 0],
           [9, 2, 0],
           [9, 3, 0],
           [9, 4, 1],
           [9, 5, 1],
           [9, 6, 1],
           [9, 7, 0],
           [9, 8, 0],
           [9, 9, 0]])




```python
# np.unique([[1, 2, 3, 4, 2, 1, 3]] * 2, return_inverse=True)[1].reshape((2, 7))
# np.unique(donut, return_inverse=True)[1].reshape((res, res))

a = np.array([1, 4, 7, 4, 2, 3, 1])
x, y = np.unique(a, return_index=True)
print(x, y)
print(a[y])
```

    [1 2 3 4 7] [0 4 5 1 2]
    [1 2 3 4 7]



```python
y.dtype
```




    dtype('int64')




```python
sliding_window_view(z := np.arange(100).reshape((10, 10)), (2, 2), axis=(0, 1))[::2, ::2].reshape((25, 2, 2))
```




    array([[[ 0,  1],
            [10, 11]],
    
           [[ 2,  3],
            [12, 13]],
    
           [[ 4,  5],
            [14, 15]],
    
           [[ 6,  7],
            [16, 17]],
    
           [[ 8,  9],
            [18, 19]],
    
           [[20, 21],
            [30, 31]],
    
           [[22, 23],
            [32, 33]],
    
           [[24, 25],
            [34, 35]],
    
           [[26, 27],
            [36, 37]],
    
           [[28, 29],
            [38, 39]],
    
           [[40, 41],
            [50, 51]],
    
           [[42, 43],
            [52, 53]],
    
           [[44, 45],
            [54, 55]],
    
           [[46, 47],
            [56, 57]],
    
           [[48, 49],
            [58, 59]],
    
           [[60, 61],
            [70, 71]],
    
           [[62, 63],
            [72, 73]],
    
           [[64, 65],
            [74, 75]],
    
           [[66, 67],
            [76, 77]],
    
           [[68, 69],
            [78, 79]],
    
           [[80, 81],
            [90, 91]],
    
           [[82, 83],
            [92, 93]],
    
           [[84, 85],
            [94, 95]],
    
           [[86, 87],
            [96, 97]],
    
           [[88, 89],
            [98, 99]]])


