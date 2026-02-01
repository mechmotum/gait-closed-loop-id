import os

import numpy as np
import matplotlib.pyplot as plt

DATAFILE = '020-longitudinal-perturbation-gait-cycles.csv'
DATAPATH = os.path.join(os.path.dirname(__file__), '..', 'data', DATAFILE)

"""
There is an h5 file for each event (First Normal Walking, Longitudinal
Pertrubation, Second Normal Walking) that includes a key `gait_cycles` which is
a Pandas Panel exported as hdf5 data. The Panel's first index indexes through
the gait cycles and selects a DataFrame that has percent gait cycle as the row
index and all the columns with marker and joint data.

Weirdly, the X, Y, Z axes do not match the coordinate system on the treadmill
in Figure 1 in my paper. There it shows the direction of travel in the -Z
direction. Maybe GTK transforms the axes and the raw data is in the Cortex
axes.
X : positive forwards
Y : postive up

In opty, we will generate joint angles for a gait cycle that can be accessed in
the form of a 2D array with (number time points, number of joint angles).

From the measurement data we can extract the same joint angles vs time or
percent gait cycle for a single gait cycle along with the belt speed.

0. major (gait cycle number)
1. minor (percent gait cycle)
2. TimeStamp
3. FrameNumber
4. LHEAD.PosX
5. LHEAD.PosY
6. LHEAD.PosZ
7. THEAD.PosX
8. THEAD.PosY
9. THEAD.PosZ
10. RHEAD.PosX
11. RHEAD.PosY
12. RHEAD.PosZ
13. FHEAD.PosX
14. FHEAD.PosY
15. FHEAD.PosZ
16.  C7.PosX
17.  C7.PosY
18.  C7.PosZ
19. T10.PosX
20. T10.PosY
21. T10.PosZ
22. SACR.PosX
23. SACR.PosY
24. SACR.PosZ
25. NAVE.PosX
26. NAVE.PosY
27. NAVE.PosZ
28. XYPH.PosX
29. XYPH.PosY
30. XYPH.PosZ
31. STRN.PosX
32. STRN.PosY
33. STRN.PosZ
34. BBAC.PosX
35. BBAC.PosY
36. BBAC.PosZ
37. LSHO.PosX
38. LSHO.PosY
39. LSHO.PosZ
40. LDELT.PosX
41. LDELT.PosY
42. LDELT.PosZ
43. LLEE.PosX
44. LLEE.PosY
45. LLEE.PosZ
46. LMEE.PosX
47. LMEE.PosY
48. LMEE.PosZ
49. LFRM.PosX
50. LFRM.PosY
51. LFRM.PosZ
52. LMW.PosX
53. LMW.PosY
54. LMW.PosZ
55. LLW.PosX
56. LLW.PosY
57. LLW.PosZ
58. LFIN.PosX
59. LFIN.PosY
60. LFIN.PosZ
61. RSHO.PosX
62. RSHO.PosY
63. RSHO.PosZ
64. RDELT.PosX
65. RDELT.PosY
66. RDELT.PosZ
67. RLEE.PosX
68. RLEE.PosY
69. RLEE.PosZ
70. RMEE.PosX
71. RMEE.PosY
72. RMEE.PosZ
73. RFRM.PosX
74. RFRM.PosY
75. RFRM.PosZ
76. RMW.PosX
77. RMW.PosY
78. RMW.PosZ
79. RLW.PosX
80. RLW.PosY
81. RLW.PosZ
82. RFIN.PosX
83. RFIN.PosY
84. RFIN.PosZ
85. LASIS.PosX
86. LASIS.PosY
87. LASIS.PosZ
88. RASIS.PosX
89. RASIS.PosY
90. RASIS.PosZ
91. LPSIS.PosX
92. LPSIS.PosY
93. LPSIS.PosZ
94. RPSIS.PosX
95. RPSIS.PosY
96. RPSIS.PosZ
97. LGTRO.PosX
98. LGTRO.PosY
99. LGTRO.PosZ
100. FLTHI.PosX
101. FLTHI.PosY
102. FLTHI.PosZ
103. LLEK.PosX
104. LLEK.PosY
105. LLEK.PosZ
106. LATI.PosX
107. LATI.PosY
108. LATI.PosZ
109. LLM.PosX
110. LLM.PosY
111. LLM.PosZ
112. LHEE.PosX
113. LHEE.PosY
114. LHEE.PosZ
115. LTOE.PosX
116. LTOE.PosY
117. LTOE.PosZ
118. LMT5.PosX
119. LMT5.PosY
120. LMT5.PosZ
121. RGTRO.PosX
122. RGTRO.PosY
123. RGTRO.PosZ
124. FRTHI.PosX
125. FRTHI.PosY
126. FRTHI.PosZ
127. RLEK.PosX
128. RLEK.PosY
129. RLEK.PosZ
130. RATI.PosX
131. RATI.PosY
132. RATI.PosZ
133. RLM.PosX
134. RLM.PosY
135. RLM.PosZ
136. RHEE.PosX
137. RHEE.PosY
138. RHEE.PosZ
139. RTOE.PosX
140. RTOE.PosY
141. RTOE.PosZ
142. RMT5.PosX
143. RMT5.PosY
144. RMT5.PosZ
145. FP1.CopX
146. FP1.CopY
147. FP1.CopZ
148. FP1.ForX
149. FP1.ForY
150. FP1.ForZ
151. FP1.MomX
152. FP1.MomY
153. FP1.MomZ
154. FP2.CopX
155. FP2.CopY
156. FP2.CopZ
157. FP2.ForX
158. FP2.ForY
159. FP2.ForZ
160. FP2.MomX
161. FP2.MomY
162. FP2.MomZ
163. F1Y1
164. F1Y2
165. F1Y3
166. F1X1
167. F1X2
168. F1Z1
169. F2Y1
170. F2Y2
171. F2Y3
172. F2X1
173. F2X2
174. F2Z1
175. Sensor13_EMG
176. Sensor13_AccX
177. Sensor13_AccY
178. Sensor13_AccZ
179. Sensor14_EMG
180. Sensor14_AccX
181. Sensor14_AccY
182. Sensor14_AccZ
183. Sensor15_EMG
184. Sensor15_AccX
185. Sensor15_AccY
186. Sensor15_AccZ
187. D-Flow Time
188. Time
189. LeftBeltSpeed
190. RightBeltSpeed
191. ReferenceFY_FP1
192. Left.Hip.Flexion.Angle
193. Left.Knee.Flexion.Angle
194. Left.Ankle.PlantarFlexion.Angle
195. Left.Hip.Flexion.Rate
196. Left.Knee.Flexion.Rate
197. Left.Ankle.PlantarFlexion.Rate
198. Left.Hip.Flexion.Moment
199. Left.Knee.Flexion.Moment
200. Left.Ankle.PlantarFlexion.Moment
201. Left.Hip.X.Force
202. Left.Hip.Y.Force
203. Left.Knee.X.Force
204. Left.Knee.Y.Force
205. Left.Ankle.X.Force
206. Left.Ankle.Y.Force
207. Right.Hip.Flexion.Angle
208. Right.Knee.Flexion.Angle
209. Right.Ankle.PlantarFlexion.Angle
210. Right.Hip.Flexion.Rate
211. Right.Knee.Flexion.Rate
212. Right.Ankle.PlantarFlexion.Rate
213. Right.Hip.Flexion.Moment
214. Right.Knee.Flexion.Moment
215. Right.Ankle.PlantarFlexion.Moment
216. Right.Hip.X.Force
217. Right.Hip.Y.Force
218. Right.Knee.X.Force
219. Right.Knee.Y.Force
220. Right.Ankle.X.Force
221. Right.Ankle.Y.Force
222. Original Time
223. Percent Gait Cycle

"""

"""
0: 222. Original Time
1: 223. Percent Gait Cycle
2: 189. LeftBeltSpeed
3: 190. RightBeltSpeed
4: 192. Left.Hip.Flexion.Angle
5: 193. Left.Knee.Flexion.Angle
6: 194. Left.Ankle.PlantarFlexion.Angle
7: 195. Left.Hip.Flexion.Rate
8: 196. Left.Knee.Flexion.Rate
9: 197. Left.Ankle.PlantarFlexion.Rate
10: 207. Right.Hip.Flexion.Angle
11: 208. Right.Knee.Flexion.Angle
12: 209. Right.Ankle.PlantarFlexion.Angle
13: 210. Right.Hip.Flexion.Rate
14: 211. Right.Knee.Flexion.Rate
15: 212. Right.Ankle.PlantarFlexion.Rate
"""

idxs = [222, 223, 189, 190, 192, 193, 194, 195, 196, 197, 207, 208, 209, 210,
        211, 212]
arr = np.loadtxt(DATAPATH, skiprows=1, delimiter=',')
get_cycle = lambda i: arr[i*20:(i + 1)*20, idxs]
cycle = get_cycle(100)

plt.plot(cycle[:, 1], cycle[:, [4, 5, 6]], color='C0')
plt.plot(cycle[:, 1], cycle[:, [10, 11, 12]], color='C1')
plt.show()
